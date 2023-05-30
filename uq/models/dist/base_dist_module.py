from abc import abstractmethod

import torch

from ..base_module import BaseModule
from ..general.mlp import MLP_MixturePrediction, MLP_SplinePrediction
from ..general.interleaved_training import compute_loss_model
from ..pred_type.recalibrated_dist import RecalibratedDist

from uq.metrics.dist_metrics_computer import DistMetricsComputer
from uq.metrics.general import nll
from uq.metrics.quantiles import crps_normal_mixture
from uq.models.general.post_hoc_calibration import (
    PostHocPitCalibration, PostHocStochasticPitCalibration, PostHocLinearPitCalibration, PostHocSmoothPitCalibration
)
from uq.utils.dist import unnormalize_dist_y


class BaseDistModule(BaseModule):
    def __init__(self, pred_type='mixture', mixture_size=3, count_bins=6, posthoc_method=None, *args, **kwargs):
        super().__init__(*args, posthoc_model=posthoc_method, **kwargs)
        self.save_hyperparameters()
        assert pred_type in ['mixture', 'spline']
        assert posthoc_method in [None, 'ecdf', 'stochastic_ecdf', 'linear_ecdf', 'smooth_ecdf']
    
    def build_model(self, **kwargs):
        homoscedastic = self.hparams.misspecification == 'homoscedasticity'
        if self.hparams.pred_type == 'mixture':
            return MLP_MixturePrediction(
                mixture_size=self.hparams.mixture_size, homoscedastic=homoscedastic, **self.get_mlp_args(), **kwargs
            )
        elif self.hparams.pred_type == 'spline':
            assert not homoscedastic
            return MLP_SplinePrediction(
                count_bins=self.hparams.count_bins, **self.get_mlp_args(), **kwargs
            )

    def compute_base_loss(self, dist, y):
        if self.hparams.base_loss == 'nll':
            loss = nll(dist, y).mean()
        elif self.hparams.base_loss == 'crps':
            loss = dist.crps(y).mean()
        else:
            raise ValueError(f'Invalid base_loss: {self.hparams.base_loss}')
        if self.hparams.misspecification == 'sharpness_reward':
            if self.hparams.pred_type == 'mixture':
                loss = loss + dist.stddev.mean() * 10.
            else:
                raise NotImplementedError('The standard deviation of a spline is not implemented')
        return loss

    def compute_metrics(self, dist, x, y, monitor_value=None, stage=None):
        # Avoid to run the model if we do not compute metrics
        computer = DistMetricsComputer(self)
        if not computer.should_compute(stage):
            return computer.compute(stage, monitor_value)
        
        if self.posthoc_model is not None:
            dist = RecalibratedDist(dist, self.posthoc_model)
        dist, y = unnormalize_dist_y(dist, y, self.scaler, self.rc)
        return DistMetricsComputer(self, y, dist).compute(stage, monitor_value)
    
    def build_posthoc_model_custom_args(self, posthoc_dataset, posthoc_method, dataset=None):
        # Note: the posthoc model can be used indifferently with both normalized and unnormalized predictions and observations.
        # The reason is that both cases will give the same PITs.
        if posthoc_dataset is None or posthoc_method is None:
            return None

        def sample(data):
            if self.hparams.posthoc_method == 'smooth_ecdf':
                # smooth_ecdf is too slow and requires a lot of memory when computed on the whole dataset.
                # Especially with smooth_ecdf, truncating this dataset makes almost no difference.
                sample_size = min(2048, len(data))
                idx = torch.distributions.Categorical(probs=torch.ones(len(data))).sample([sample_size])
                x, y = data[:]
                return x[idx], y[idx]
            return data[:]

        # When posthoc_dataset == 'batch', the post-hoc model is computed on half the batch during training 
        # but on the calibration dataset during validation and testing
        if posthoc_dataset in ['calib', 'batch']:
            dataset = sample(self.trainer.datamodule.data_calib)
        elif posthoc_dataset == 'train':
            dataset = sample(self.trainer.datamodule.data_train)
        else:
            # Else, the dataset is directly given as argument
            pass
        x, y = dataset
        pits = self.model.dist(x).cdf(y.squeeze(dim=-1))
        if posthoc_method == 'ecdf':
            return PostHocPitCalibration(pits)
        elif posthoc_method == 'stochastic_ecdf':
            return PostHocStochasticPitCalibration(pits)
        elif posthoc_method == 'linear_ecdf':
            return PostHocLinearPitCalibration(pits)
        elif posthoc_method == 'smooth_ecdf':
            return PostHocSmoothPitCalibration(pits, model=self)
    
    def build_posthoc_model(self, dataset=None):
        return self.build_posthoc_model_custom_args(self.hparams.posthoc_dataset, self.hparams.posthoc_method, dataset=dataset)


class DistModule(BaseDistModule):
    def __init__(self, *args, lambda_=0, interleaved=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @abstractmethod
    def compute_regul(self, dist, y):
        pass

    def compute_loss(self, dist, y):
        return compute_loss_model(self, dist, y)
    
    def predict(self, x):
        return self.model.dist(x)
    
    def step(self, batch, stage):
        x, y = batch
        y = y.squeeze(dim=-1)

        dist = self.predict(x)
        loss, base_loss, regul = self.compute_loss(dist, y)

        with torch.no_grad():
            metrics = self.compute_metrics(dist, x, y, monitor_value=base_loss.detach(), stage=stage)
        metrics['loss'] = loss
        metrics['base_loss'] = base_loss
        if self.hparams.lambda_ != 0:
            metrics['regul'] = regul.detach()
        return metrics
