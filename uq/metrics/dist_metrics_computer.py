import torch

from uq.models.regul.marginal_regul import qr_loss
from uq.utils.dist import icdf

from .calibration import get_observed_frequency, quantile_calibration_from_pits_with_sorting
from .general import nll
from .independence import delta_ils_coverage_from_models, indep_of_length_and_coverage_pearson
from .metrics_computer import MetricsComputer
from .quantiles import length_and_coverage_from_quantiles, quantile_scores, wis


class DistMetricsComputer(MetricsComputer):
    def __init__(self, module, y=None, dist=None, **kwargs):
        super().__init__(module, **kwargs)
        self.y = y
        self.dist = dist

    def monitored_metrics(self):
        nll_value = nll(self.dist, self.y).mean()
        return {
            'nll': nll_value,
        }

    def train_metrics(self):
        nll_value = nll(self.dist, self.y).mean()
        pits = self.dist.cdf(self.y)
        calib_l1 = quantile_calibration_from_pits_with_sorting(pits, L=1)
        calib_l2 = quantile_calibration_from_pits_with_sorting(pits, L=2)
        calib_kl = qr_loss(self.dist, self.y, neural_sort=False)

        alpha = torch.arange(0.05, 1, 0.05)
        quantiles = icdf(self.dist, alpha[:, None]).permute(1, 0)
        assert quantiles.shape == self.dist.batch_shape + alpha.shape
        quantile_scores_values = quantile_scores(self.y, quantiles, alpha)
        quantile_scores_per_level = quantile_scores_values.mean(dim=0)
        # We just take the mean for all coverage levels (same weight for each coverage level)
        pearson = indep_of_length_and_coverage_pearson(self.y, quantiles).mean()
        wis_value = wis(quantile_scores_values)
        try:
            crps = self.dist.crps(self.y)
        except (NotImplementedError, AttributeError):
            crps = torch.full_like(self.y, torch.nan)

        quantiles_scores = {
            f'quantile_score_{level:.2f}': score for level, score in zip(alpha, quantile_scores_per_level)
        }

        observed_frequency = get_observed_frequency(pits, alpha)
        observed_frequency_metrics = {
            f'observed_frequency_{level:.2f}': value for level, value in zip(alpha, observed_frequency)
        }

        length_90, coverage_90 = length_and_coverage_from_quantiles(self.y, quantiles, alpha, 0.05, 0.95)

        compute_mean_from_samples = False
        compute_stddev_from_samples = False
        try:
            mean = self.dist.mean
        except NotImplementedError:
            compute_mean_from_samples = True
        try:
            stddev = self.dist.stddev
        except NotImplementedError:
            compute_stddev_from_samples = True
        # Only compute samples if necessary
        if compute_mean_from_samples or compute_stddev_from_samples:
            if compute_mean_from_samples:
                mean = quantiles.mean(dim=-1)
            if compute_stddev_from_samples:
                stddev = quantiles.std(dim=-1)

        median_index = (alpha == 0.5).nonzero().item()
        median = quantiles[..., median_index]

        return {
            'nll': nll_value,
            'quantile_score': quantile_scores_per_level.mean(),
            'calib_l1': calib_l1,
            'calib_l2': calib_l2,
            'calib_kl': calib_kl,
            'wis': wis_value.mean(),
            'crps': crps.mean(),
            'mean': mean.mean(),
            'stddev': stddev.mean(),
            'var': (stddev**2).mean(),
            'pearson': pearson,
            'length_90': length_90.mean(),
            'coverage_90': coverage_90.mean(),
            'rmse': (mean - self.y).square().mean().sqrt(),
            'mae': (median - self.y).abs().mean(),
            **quantiles_scores,
            **observed_frequency_metrics,
        }

    def val_metrics(self):
        return self.train_metrics()

    def test_metrics(self):
        return self.val_metrics()
