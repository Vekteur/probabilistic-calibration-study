import torch

from .base_dist_module import DistModule
from ..general.post_hoc_calibration import PostHocPitCalibration
from ..pred_type.recalibrated_dist import RecalibratedDist
from uq.utils.dist import unnormalize_dist_y


class DistRecalibrated(DistModule):
    def __init__(self, *args, recal_dataset='train', **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def step(self, batch, stage):
        x, y = batch
        y = y.squeeze(dim=-1)

        dist = self.predict(x)
        with torch.no_grad():
            if self.hparams.recal_dataset == 'calib':
                calib_x, calib_y = self.trainer.datamodule.data_calib[:]
                pit = self.model.dist(calib_x).cdf(calib_y.squeeze(dim=1))
            else:
                pit = dist.cdf(y)
            posthoc_model = PostHocPitCalibration(pit)
        dist = RecalibratedDist(dist, posthoc_model)
        loss, base_loss, regul = self.compute_loss(dist, y)

        with torch.no_grad():
            dist, y = unnormalize_dist_y(dist, y, self.scaler, self.rc)
            metrics = self.compute_metrics(dist, x, y, monitor_value=base_loss.detach(), stage=stage)
        metrics["loss"] = loss
        metrics["base_loss"] = base_loss
        if self.hparams.lambda_ != 0:
            metrics["regul"] = regul.detach()
        return metrics
