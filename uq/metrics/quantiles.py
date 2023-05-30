import torch

from uq import utils

log = utils.get_logger(__name__)


def quantile_scores(y, quantiles, alpha):
    """
    Return the quantile score for a list of quantile levels.
    Taking the mean of this score over the levels will give a proper scoring rule for each levels.
    """
    batch_size, n_levels = quantiles.shape
    if alpha.dim() == 1:   # If alpha is the same for each elements of the batch
        alpha = alpha[None, :]   # We add the batch dimension
    assert alpha.shape[-1] == n_levels and y.shape == (batch_size,)
    diff = y[:, None] - quantiles
    indicator = (diff < 0).float()
    score_per_quantile = diff * (alpha - indicator)
    return score_per_quantile


def interval_scores(quantile_scores, alpha):
    """
    It is supposed that quantile_scores[i] and quantile_scores[-i] have corresponding levels alpha/2 and 1-alpha/2
    """
    mid = len(alpha) / 2
    assert alpha[:mid] == 1 - alpha[mid + 1 :: -1]
    coverage = 1 - 2 * alpha[:mid]
    return 2 * (quantile_scores[:, :mid] + quantile_scores[:, mid + 1 :: -1]) / coverage


def wis(quantile_scores):
    """
    Args:
        quantile_scores: tensor that contains the quantile scores.
        It is supposed that the corresponding quantile levels are [l_1, l_2, ..., l_k]
        and that l_1 - 0 == l_2 - l_1 == ... == 1 - l_k.
    Returns:
        The weighted interval score, an approximation of the CRPS.
    """
    side_values = (quantile_scores[:, :1] + quantile_scores[:, -1:]) * 3 / 2
    quantile_scores = torch.cat((quantile_scores, side_values), dim=1)
    return 2 * quantile_scores.mean(dim=1)


def crps_helper(mean, std):
    dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(mean))
    return mean * (2 * dist.cdf(mean / std) - 1) + 2 * std * torch.exp(dist.log_prob(mean / std))


def crps_normal_mixture_from_params(mean, std, w, y):
    batch_size, mixture_size = mean.shape
    term1 = w * crps_helper(y[:, None] - mean, std)
    var = std**2
    factor = crps_helper(
        mean[:, :, None] - mean[:, None, :],
        torch.sqrt(
            var[:, :, None] + var[:, None, :] + 1e-12
        ),  # The derivative of torch.sqrt is not defined in 0
    )
    term2 = w[:, :, None] * w[:, None, :] * factor
    return term1.sum(dim=1) - term2.sum(dim=(1, 2)) / 2


def crps_normal_mixture(dist, y):
    return crps_normal_mixture_from_params(
        dist.component_distribution.loc,
        dist.component_distribution.scale,
        dist.mixture_distribution.probs,
        y,
    )


def length_and_coverage_from_quantiles(y, quantiles, alpha, left_alpha, right_alpha):
    left_alpha_index = (alpha == left_alpha).nonzero().item()
    right_alpha_index = (alpha == right_alpha).nonzero().item()
    left_bound = quantiles[..., left_alpha_index]
    right_bound = quantiles[..., right_alpha_index]
    length = torch.maximum(right_bound - left_bound, torch.tensor(0))
    coverage = ((left_bound < y) & (y < right_bound)).float()
    return length, coverage


def quantile_sharpness_reward(quantiles, alpha):
    """
    The quantile sharpness reward corresponds to the
    """
    mid = quantiles.shape[1] // 2
    assert alpha[:mid] == 1 - alpha[mid + 1 :: -1]
    miscoverage = 2 * alpha[:mid]
    left_bound = quantiles[..., :mid]
    right_bound = quantiles[..., mid:].flip(dims=(-1,))
    length = torch.maximum(right_bound - left_bound, torch.tensor(0))
    weighted_length = miscoverage / 2.0 * length
    return weighted_length.mean(dim=-1)
