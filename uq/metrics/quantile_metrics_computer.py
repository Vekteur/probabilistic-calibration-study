import torch
from scipy.interpolate import interp1d

from uq.models.general.post_hoc_calibration import PostHocConformalCalibration
from uq.utils.torch_utils import centered_bins

from .calibration import get_observed_frequency, quantile_calibration_from_quantiles
from .general import nll
from .independence import delta_ils_coverage_from_models, indep_of_length_and_coverage_pearson
from .metrics_computer import MetricsComputer
from .quantiles import length_and_coverage_from_quantiles, quantile_scores, wis


class QuantileMetricsComputer(MetricsComputer):
    def __init__(self, module, y=None, quantiles=None, alpha=None, **kwargs):
        super().__init__(module, **kwargs)
        self.y = y
        self.quantiles = quantiles
        self.alpha = alpha

    def monitored_metrics(self):
        quantile_scores_values = quantile_scores(self.y, self.quantiles, self.alpha)
        quantile_scores_per_level = quantile_scores_values.mean(dim=0)
        return {
            'quantile_score': quantile_scores_per_level.mean(),
        }

    def train_metrics(self):
        quantile_scores_values = quantile_scores(self.y, self.quantiles, self.alpha)
        quantile_scores_per_level = quantile_scores_values.mean(dim=0)

        calib_l1 = quantile_calibration_from_quantiles(self.quantiles, self.y, L=1, alpha=self.alpha)
        calib_l2 = quantile_calibration_from_quantiles(self.quantiles, self.y, L=2, alpha=self.alpha)

        pearson = indep_of_length_and_coverage_pearson(self.y, self.quantiles).mean()
        wis_value = wis(quantile_scores_values)
        samples = self.quantiles
        mean = samples.mean(dim=1)
        stddev = samples.std(dim=1)

        quantiles_scores = {
            f'quantile_score_{level:.2f}': score
            for level, score in zip(self.alpha, quantile_scores_per_level)
        }
        observed_frequency = get_observed_frequency(self.y, self.quantiles)
        observed_frequency_metrics = {
            f'observed_frequency_{level:.2f}': value for level, value in zip(self.alpha, observed_frequency)
        }

        length_90, coverage_90 = length_and_coverage_from_quantiles(
            self.y, self.quantiles, self.alpha, 0.05, 0.95
        )

        median_index = (self.alpha == 0.5).nonzero().item()
        median = self.quantiles[..., median_index]

        return {
            'expected_qs': quantile_scores_per_level.mean(),
            'calib_l1': calib_l1,
            'calib_l2': calib_l2,
            'wis': wis_value.mean(),
            'mean': mean.mean(),
            'stddev': stddev.mean(),
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
