from abc import abstractmethod

import torch
from pytorch_lightning import LightningModule

from uq.metrics.metrics_collector import MetricsCollector
from uq.utils.general import elapsed_timer
from uq.utils.hparams import Choice, Join, Union

# posthoc_method: the p
#
#


class BaseModule(LightningModule):
    def __init__(
        self,
        rc=None,
        input_size=None,
        lr=1e-2,
        nb_hidden=3,
        units_size=128,
        misspecification=None,
        drop_prob=0.2,
        base_loss=None,
        posthoc_model=None,
        posthoc_dataset=None,
        batch_size=None,
    ):
        """
        Args:
            posthoc_dataset: The dataset on which the posthoc model is computed.
            posthoc_model: The mapping that transforms predictions. It is computed at the start of
                each training batch, validation epoch, or test epoch. Enabling the computation of
                training metrics can have a bad impact on computation time.
        """

        super().__init__()
        self.save_hyperparameters(ignore='rc')
        self.rc = rc
        assert self.hparams.misspecification in [
            None,
            'small_mlp',
            'big_mlp',
            'homoscedasticity',
            'sharpness_reward',
        ]
        assert self.hparams.posthoc_dataset in [
            None,
            'calib',
            'train',
            'batch',
        ]
        assert (self.hparams.posthoc_model is None) == (self.hparams.posthoc_dataset is None)
        if self.hparams.misspecification == 'small_mlp':
            self.hparams.nb_hidden = 1
            self.hparams.units_size = 32
        elif self.hparams.misspecification == 'big_mlp':
            self.hparams.nb_hidden = 30
        self.randomized_predictions = False
        self.model = self.build_model()
        self.metrics_collector = MetricsCollector(self)

    @property
    def monitor(self):
        return 'base_loss'

    def get_mlp_args(self):
        return dict(
            input_size=self.hparams.input_size,
            hidden_sizes=[self.hparams.units_size for _ in range(self.hparams.nb_hidden)],
            drop_prob=self.hparams.drop_prob,
        )

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def step(self, batch, stage):
        pass

    def timed_step(self, batch, stage):
        with elapsed_timer() as time:
            result = self.step(batch, stage)
        self.metrics_collector.advance_timer(f'{stage}_time', time())
        return result

    @abstractmethod
    def build_posthoc_model(self, dataset=None):
        pass

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        return self(batch)

    def training_step(self, batch, batch_idx):
        if self.rc.config.save_train_metrics:
            if self.hparams.posthoc_dataset == 'batch':
                x, y = batch
                batch_size = x.shape[0]
                train_batch_size = batch_size // 2
                posthoc_batch = (x[train_batch_size:], y[train_batch_size:])
                self.posthoc_model = self.build_posthoc_model(posthoc_batch)
                batch = batch[:train_batch_size]
            else:
                self.posthoc_model = self.build_posthoc_model()
        metrics = self.timed_step(batch, 'train')
        if self.rc.config.save_train_metrics:
            self.log(
                f'train/{self.monitor}',
                metrics[self.monitor],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return metrics

    def validation_step(self, batch, batch_idx):
        metrics = self.timed_step(batch, 'val')
        self.log(
            f'val/{self.monitor}',
            metrics[self.monitor],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return metrics

    def test_step(self, batch, batch_idx: int):
        return self.timed_step(batch, 'test')

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    def on_train_start(self):
        if self.hparams.batch_size is not None:
            self.trainer.datamodule.hparams.batch_size = self.hparams.batch_size
        self.scaler = self.trainer.datamodule.scaler_y

    def training_epoch_end(self, outputs):
        self.metrics_collector.collect_per_step(outputs, 'train')

    def on_validation_start(self):
        if self.rc.config.save_val_metrics:
            self.posthoc_model = self.build_posthoc_model()

    def validation_epoch_end(self, outputs):
        self.metrics_collector.collect_per_step(outputs, 'val')
        self.metrics_collector.add_best_iter_metrics()

    def on_test_start(self):
        if self.rc.config.save_test_metrics:
            self.posthoc_model = self.build_posthoc_model()

    def test_epoch_end(self, outputs):
        self.metrics_collector.collect_per_step(outputs, 'test')
