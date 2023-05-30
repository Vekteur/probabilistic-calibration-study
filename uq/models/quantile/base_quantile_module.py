from abc import abstractmethod

import torch

from uq.metrics.quantile_metrics_computer import MetricsComputer, QuantileMetricsComputer
from uq.metrics.quantiles import interval_scores, quantile_scores, quantile_sharpness_reward
from uq.models.general.post_hoc_calibration import PostHocConformalCalibration
from uq.utils.dist import unnormalize_quantiles

from ..base_module import BaseModule
from ..general.interleaved_training import compute_loss_model
from ..general.mlp import MLP_QuantilePrediction, MLP_QuantilePredictionIndependent


class BaseQuantileModule(BaseModule):
    def __init__(
        self,
        pred_type='quantile',
        n_quantiles=1,
        base_loss='expected_qs',
        independent_alpha=True,
        monotonic=False,
        posthoc_method=None,
        *args,
        **kwargs
    ):
        super().__init__(*args, posthoc_model=posthoc_method, **kwargs)
        self.save_hyperparameters()
        assert pred_type in ['quantile']
        assert posthoc_method in [None, 'CQR']
        assert (
            self.hparams.misspecification != 'homoscedasticity'
        ), 'Homoscedasticity is not implemented for quantile models'

    def build_model(self):
        kwargs = dict(monotonic=self.hparams.monotonic, **self.get_mlp_args())
        if self.hparams.independent_alpha:
            return MLP_QuantilePredictionIndependent(**kwargs)
        else:
            return MLP_QuantilePrediction(n_quantiles=self.hparams.n_quantiles, **kwargs)

    def compute_base_loss(self, quantiles, y, alpha):
        if self.hparams.base_loss == 'expected_qs':
            loss = quantile_scores(y, quantiles, alpha=alpha).mean()
        else:
            raise ValueError('Invalid base_loss')

        if self.hparams.misspecification == 'sharpness_reward':
            assert self.hparams.pred_type == 'quantile'
            loss = loss + quantile_sharpness_reward(quantiles, alpha)
        return loss

    @property
    def alpha_for_metrics(self):
        return torch.arange(0.05, 1, 0.05)

    def compute_metrics(self, model, x, y, monitor_value=None, stage=None):
        # Avoid to run the model if we do not compute metrics
        computer = QuantileMetricsComputer(self)
        if not computer.should_compute(stage):
            return computer.compute(stage, monitor_value)

        alpha = self.alpha_for_metrics
        # For metrics such as quantile calibration, we need quantiles at the same level
        # However, during training, quantiles should be ideally at random levels
        # Thus, we compute metrics on new predictions on fixed levels
        quantiles = model.quantiles(x, alpha)
        quantiles, y = unnormalize_quantiles(quantiles, y, self.scaler, self.rc)
        if self.posthoc_model is not None:
            quantiles = self.posthoc_model.transform(quantiles)
        return QuantileMetricsComputer(self, y, quantiles, alpha).compute(stage, monitor_value)

    def build_posthoc_model_custom_args(self, posthoc_dataset, posthoc_method):
        if posthoc_dataset is None or posthoc_method is None:
            return None

        if posthoc_dataset in ['calib', 'batch']:
            dataset = self.trainer.datamodule.data_calib[:]
        elif posthoc_dataset == 'train':
            dataset = self.trainer.datamodule.data_train[:]
        else:
            # Else, the dataset is directly given as argument
            pass

        alpha = self.alpha_for_metrics
        x, y = dataset
        quantiles = self.model.quantiles(x, alpha)
        if posthoc_method == 'CQR':
            return PostHocConformalCalibration(y, quantiles, alpha, scaler=self.scaler, rc=self.rc)

    def build_posthoc_model(self):
        return self.build_posthoc_model_custom_args(self.hparams.posthoc_dataset, self.hparams.posthoc_method)


class QuantileModule(BaseQuantileModule):
    def __init__(self, *args, lambda_=0, interleaved=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    @abstractmethod
    def compute_regul(self, quantiles, y, alpha):
        pass

    def compute_loss(self, quantiles, y, alpha):
        return compute_loss_model(self, quantiles, y, alpha)

    def predict(self, x, alpha):
        return self.model(x, alpha)

    def step(self, batch, stage):
        x, y = batch
        y = y.squeeze(dim=-1)

        # We estimate the same quantiles for each element of the batch
        # It is necessary because we need to have the same quantile levels for regularization
        # alpha = torch.rand(x.shape[0], self.hparams.n_quantiles, device=x.device)
        alpha = torch.rand(self.hparams.n_quantiles, device=x.device)
        alpha = torch.sort(alpha)[0]
        quantiles = self.predict(x, alpha)
        loss, base_loss, regul = self.compute_loss(quantiles, y, alpha)

        with torch.no_grad():
            metrics = self.compute_metrics(self.model, x, y, monitor_value=base_loss.detach(), stage=stage)
        metrics['loss'] = loss
        metrics['base_loss'] = base_loss
        if self.hparams.lambda_ != 0:
            metrics['regul'] = regul.detach()
        return metrics
