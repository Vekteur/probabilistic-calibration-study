import torch

from uq.utils.dist import icdf


def coverage(l, y, u):
    return ((l < y) & (y < u)).float()


# TODO


def indep_of_length_and_coverage_pearson(y, quantiles):
    half = quantiles.shape[1] // 2
    lower_bounds = quantiles[:, :half]
    upper_bounds = quantiles[:, -half:].flip(dims=(1,))
    interval_lengths = upper_bounds - lower_bounds
    interval_coverages = coverage(lower_bounds, y[:, None], upper_bounds)
    return pearson_corr(interval_lengths, interval_coverages, dim=0).abs()


def pearson_corr(x, y, dim=-1):
    """
    Compute the Pearson correlation coefficient along the dimension `dim`.
    """
    vx = x - torch.mean(x, dim=dim, keepdim=True)
    vy = y - torch.mean(y, dim=dim, keepdim=True)

    # Don't take the square root of 0 because its derivative is nan in PyTorch
    denom = torch.sqrt(torch.sum(vx**2, dim=dim) + 1e-12) * torch.sqrt(torch.sum(vy**2, dim=dim) + 1e-12)
    return torch.sum(vx * vy, dim=dim) / (denom + 1e-6)


def pairwise_distances(x):
    # x should be 2-dimensional
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def length_and_coverage(dist, y):
    levels = torch.arange(0.01, 1, 0.01)
    quantiles = icdf(dist, levels)
    half = quantiles.shape[1] // 2
    lower_bounds = quantiles[:, :half]
    upper_bounds = quantiles[:, -half:].flip(dims=(1,))
    interval_lengths = upper_bounds - lower_bounds
    interval_coverages = coverage(lower_bounds, y[:, None], upper_bounds)
    return interval_lengths, interval_coverages


def delta_ils_coverage_from_models(dist_model1, dist_model2, y):
    length_model1, coverage_model1 = length_and_coverage(dist_model1, y)
    length_model2, coverage_model2 = length_and_coverage(dist_model2, y)
    return delta_ils_coverage(length_model1, length_model2, coverage_model1, coverage_model2)


def delta_ils_coverage(
    length_model1,
    length_model2,
    coverage_model1,
    coverage_model2,
    size_ratio=0.1,
):
    """
    Let ILS be the samples corresponding to predictions intervals that
    are larger for model1. The delta ILS coverage measures the difference of
    coverage between ILS samples and all samples.
    """
    N, nb_levels = length_model1.shape
    diff = length_model1 - length_model2
    _, sorted_diff_indices = torch.sort(diff, dim=0)
    size = int(N * size_ratio)
    selected_indices = sorted_diff_indices[-size:, :]

    delta_ils_coverage_model1 = delta_ils_coverage_helper_for_model(coverage_model1, selected_indices)
    delta_ils_coverage_model2 = delta_ils_coverage_helper_for_model(coverage_model2, selected_indices)
    return delta_ils_coverage_model1, delta_ils_coverage_model2


def delta_ils_coverage_helper_for_model(coverage, indices):
    return (coverage.gather(dim=0, index=indices).mean(dim=0) - coverage.mean(dim=0)).abs()
