import logging
import math
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

log = logging.getLogger('uq')


# Source: https://gist.github.com/farahmand-m/8a416f33a27d73a149f92ce4708beb40
class StandardScaler:
    def __init__(self, mean=None, scale=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        """
        self.mean_ = mean
        self.scale_ = scale
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean_ = torch.mean(values, dim=dims)
        self.scale_ = torch.std(values, dim=dims)
        return self

    def transform(self, values):
        return (values - self.mean_) / (self.scale_ + self.epsilon)

    def inverse_transform(self, values):
        return values * self.scale_ + self.mean_


class ScaledDataset(Dataset):
    def __init__(self, dataset, scaler_x, scaler_y):
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def scale(self, v, scaler):
        shape = v.shape
        if len(shape) == 1:
            v = v[None, :]
        scaled = scaler.transform(v)
        return scaled.reshape(shape)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = self.scale(x, self.scaler_x)
        y = self.scale(y, self.scaler_y)
        return x, y


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        rc=None,
        name=None,
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        train_inter_val_calib_test_split_ratio=None,
        seed=0,
    ):
        super().__init__()

        # This line allows to access `__init__` arguments with `self.hparams` attribute
        self.save_hyperparameters(logger=False, ignore='rc')
        self.rc = rc

    @abstractmethod
    def get_data(self):
        pass

    def make_scaled_dataset(self, ds):
        return ScaledDataset(ds, self.scaler_x, self.scaler_y)

    def get_pretrained_model(self):
        from uq.analysis.dataframes import load_config
        from uq.utils.checkpoints import load_model_checkpoint, load_rc_checkpoint

        # We assume that the experiment creating the pretrained model has been logged at the same log_base_dir than the current experiment
        log_path = Path(self.rc.config.log_base_dir) / 'save_all'
        model_rc = load_rc_checkpoint(
            config=load_config(log_path),
            dataset_group=self.rc.dataset_group,
            dataset=self.rc.dataset,
            model='no_regul',
            hparams=dict(
                nb_hidden=3,
                pred_type='mixture',
                mixture_size=5,
                base_loss='nll',
            ),
            model_cls='dist_no_regul',
        )
        model = load_model_checkpoint(model_rc).model
        model.eval()
        return model

    def create_known_uncertainty(self, x, y):
        # Ideally, we should use the original scalers but using an approximation does
        # not invalidate the results because we just create a new dataset
        scaler_x = StandardScaler().fit(x)
        scaler_y = StandardScaler().fit(y)
        # Since the model has been trained on a scaled x, we need to scale it too
        x = scaler_x.transform(x)

        # Load a previous model that was trained to predict the distribution of a scaled y
        model = self.get_pretrained_model()
        with torch.no_grad():
            cond_dist = model.dist(x)
        # Unscale the distribution so that its scale corresponds to the scale of y
        return cond_dist.unnormalize(scaler_y)

    def get_uncertainty(self, x):
        model = self.get_pretrained_model()
        with torch.no_grad():
            cond_dist = model.dist(x)
        return cond_dist.unnormalize(self.scaler_y)

    def use_known_uncertainty(self):
        return self.rc.config.use_known_uncertainty

    def subsample(self, x, y, max_size=50000):
        N = x.shape[0]
        rng = np.random.RandomState(self.hparams.seed + 1)
        train_ratio = self.hparams.train_inter_val_calib_test_split_ratio[0]
        sample_idx = rng.choice(N, min(N, math.ceil(max_size / train_ratio)), replace=False)
        return x[sample_idx], y[sample_idx]

    def load_datasets(self):
        x, y = self.get_data()
        x = torch.from_numpy(x).to(torch.float32)
        y = torch.from_numpy(y).to(torch.float32)
        if self.use_known_uncertainty():
            self.cond_dist = self.create_known_uncertainty(x, y)
            y = self.cond_dist.sample().unsqueeze(-1)
        x, y = self.subsample(x, y)   # TODO: KEEP IN MIND
        tensor_data = TensorDataset(x, y)
        self.total_size = len(tensor_data)

        # Convert ratios to number of elements in the dataset
        splits_size = (
            np.array(self.hparams.train_inter_val_calib_test_split_ratio) * len(tensor_data)
        ).astype(int)
        splits_size[-1] = len(tensor_data) - splits_size[:-1].sum()

        (self.data_train, self.data_inter, self.data_val, self.data_calib, self.data_test,) = random_split(
            dataset=tensor_data,
            lengths=splits_size.tolist(),
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )

        x, y = self.data_train[:]
        self.scaler_x = StandardScaler().fit(x)
        self.scaler_y = StandardScaler().fit(y)

        if self.rc.config.normalize:
            self.data_train = self.make_scaled_dataset(self.data_train)
            self.data_inter = self.make_scaled_dataset(self.data_inter)
            self.data_val = self.make_scaled_dataset(self.data_val)
            self.data_calib = self.make_scaled_dataset(self.data_calib)
            self.data_test = self.make_scaled_dataset(self.data_test)

        if self.use_known_uncertainty():
            x_val, y_val = self.data_val[:]
            self.cond_dist_val = self.get_uncertainty(x_val)

    def setup(self, stage):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_calib`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        if stage == 'fit':
            self.load_datasets()
        # Make the size of the inputs accessible to the models
        first_x, first_y = self.data_train[0]
        self.input_size = first_x.shape[0]

    def get_dataloader(self, dataset, drop_last=False, shuffle=False):
        return DataLoader(
            dataset=dataset,
            batch_size=min(len(dataset), self.hparams.batch_size),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def train_dataloader_interleaved(self):
        if self.trainer.current_epoch % 2 == 1:   # Training dataset
            return self.get_dataloader(self.data_train, drop_last=True, shuffle=True)
        else:   # Interleaved dataset
            return self.get_dataloader(self.data_inter, drop_last=True, shuffle=True)

    def train_dataloader(self):
        """
        Return the DataLoader of the training dataset.
        It changes depending on:
        - Whether posthoc with a calibration dataset is enabled
        - Whether interleaved training is enabled
        """
        model = self.trainer.model
        interleaved = self.trainer.model.hparams.get('interleaved')
        posthoc_calib = model.hparams.get('posthoc_dataset') == 'calib'
        if interleaved:
            data = self.train_dataloader_interleaved()
        else:
            data = torch.utils.data.ConcatDataset([self.data_train, self.data_inter])
        if not posthoc_calib:
            data = torch.utils.data.ConcatDataset([data, self.data_calib])
        # We can just drop the last batch because shuffle is set to true.
        # The advantage is that each batch is of the same size during training,
        # and leads to the same variance in the metrics.
        # Even if more epochs is required when drop_last=True, shuffled data is used in both cases.
        return self.get_dataloader(data, drop_last=True, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test)
