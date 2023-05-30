import torch

from .base_dist_module import DistModule
from ..regul.marginal_regul import cdf_based_regul_from_pit, qr_loss, quantile_based_regul


class DistEntropyRegul(DistModule):
    def __init__(self, *args, spacing=64, neural_sort=False, s=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, dist, y):
        return qr_loss(dist, y, spacing=self.hparams.spacing, neural_sort=self.hparams.neural_sort, s=self.hparams.s)


class DistCDF_Regul(DistModule):
    def __init__(self, *args, L=2, s=100., **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, dist, y):
        return cdf_based_regul_from_pit(dist.cdf(y), self.hparams.L, s=self.hparams.s)


class DistQuantileRegul(DistModule):
    def __init__(self, *args, L=2, neural_sort=False, s=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, dist, y):
        return quantile_based_regul(dist.cdf(y), self.hparams.L, neural_sort=self.hparams.neural_sort, s=self.hparams.s)
