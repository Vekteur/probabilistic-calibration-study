import torch

from uq.models.regul.oqr_regul import oqr_loss

from .base_quantile_module import QuantileModule


class QuantileOQR_Regul(QuantileModule):
    def __init__(self, *args, s=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, quantiles, y, alpha):
        return oqr_loss(quantiles, y, s=self.hparams.s)
