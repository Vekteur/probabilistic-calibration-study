from ..regul.marginal_regul import cdf_based_regul_from_quantiles, qd_regul, truncated_dist_regul
from .base_quantile_module import QuantileModule


class QuantileCDFRegul(QuantileModule):
    def __init__(self, *args, L=2, s=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, quantiles, y, alpha):
        return cdf_based_regul_from_quantiles(quantiles, y, L=self.hparams.L, alpha=alpha, s=self.hparams.s)


class TruncatedDistRegul(QuantileModule):
    def compute_regul(self, quantiles, y, alpha):
        return truncated_dist_regul(quantiles, y, alpha)


class HQPIRegul(QuantileModule):
    def __init__(self, *args, s=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

    def compute_regul(self, quantiles, y, alpha):
        return qd_regul(quantiles, y, alpha, s=self.hparams.s)
