import torch

from uq.utils.dist import unnormalize_quantiles

from ..general.post_hoc_calibration import PostHocConformalCalibration
from .base_quantile_module import QuantileModule


class QuantileRecalibrated(QuantileModule):
    def __init__(self, *args, recal_dataset='train', **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def step(self, batch, stage):
        x, y = batch
        y = y.squeeze(dim=-1)

        alpha = torch.rand(self.hparams.n_quantiles, device=x.device)
        alpha = torch.sort(alpha)[0]
        quantiles = self.predict(x, alpha)
        loss1, base_loss1, regul1 = self.compute_loss(quantiles, y, alpha)

        with torch.no_grad():
            if self.hparams.recal_dataset == 'calib':
                calib_x, calib_y = self.trainer.datamodule.data_calib[:]
                calib_quantiles = self.predict(calib_x, alpha)
                posthoc_model = PostHocConformalCalibration(calib_y, calib_quantiles, alpha)
            else:
                posthoc_model = PostHocConformalCalibration(y[:, None], quantiles, alpha)
        quantiles = posthoc_model.transform(quantiles)

        loss2, base_loss2, regul2 = self.compute_loss(quantiles, y, alpha)
        loss = loss1 + 0.001 * loss2
        base_loss = base_loss1 + 0.001 * base_loss2
        if self.hparams.lambda_ != 0:
            regul = regul1 + 0.001 * regul2

        with torch.no_grad():
            quantiles, y = unnormalize_quantiles(quantiles, y, self.scaler, self.rc)
            metrics = self.compute_metrics(self.model, x, y, monitor_value=base_loss.detach(), stage=stage)
        metrics['loss'] = loss
        metrics['base_loss'] = base_loss
        if self.hparams.lambda_ != 0:
            metrics['regul'] = regul.detach()
        return metrics
