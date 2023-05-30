import numpy as np
import torch

from uq.models.regul.marginal_regul import (
    cdf_based_regul_from_pit,
    cdf_based_regul_from_quantiles,
    quantile_based_regul,
)
from uq.utils.torch_utils import centered_bins


def get_observed_frequency(pit, alpha):
    indicator = pit[:, None] <= alpha
    return indicator.float().mean(dim=0)


def quantile_calibration_from_pits(pit, L, alpha):
    return cdf_based_regul_from_pit(pit, L, alpha, s=None)


def quantile_calibration_from_pits_with_sorting(pit, L):
    return quantile_based_regul(pit, L, neural_sort=False)


def quantile_calibration_from_quantiles(quantiles, y, L, alpha):
    return cdf_based_regul_from_quantiles(quantiles, y, L, alpha)
