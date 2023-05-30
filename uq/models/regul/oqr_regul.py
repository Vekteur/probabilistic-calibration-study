import torch

from uq.metrics.independence import pearson_corr


def relaxed_indicator(a, b, c, s=50):
    """
    Indicator function that is relaxed so that it can be backpropagated through.
    indicator(a <= b <= c) ~= relaxed_indicator(a, b, c)
    """
    return torch.sigmoid(s * torch.min(b - a, c - b))


def relaxed_backprop_indicator(a, b, c, s=1.0):
    """Indicator function that is exact but can be backpropagated through using a relaxation
    of this function.
    """
    return (
        relaxed_indicator(a, b, c, s=s)
        - relaxed_indicator(a, b, c, s=s).detach()
        + ((a < b) & (b < c)).float()
    )


def oqr_loss(quantiles, y, s=50):
    mid = quantiles.shape[1] // 2
    left_bounds = quantiles[:, :mid]
    right_bounds = quantiles[:, mid:].flip(dims=(1,))
    interval_lengths = right_bounds - left_bounds
    interval_coverages = relaxed_indicator(left_bounds, y[:, None], right_bounds, s=s)
    interval_lengths += 1e-4
    # print(interval_lengths.mean(dim=0))
    # print(interval_coverages.mean(dim=0))

    # This part of the code seems not necessary
    # interval_ = interval_coverages.min(dim=1)[0] - interval_coverages.max(dim=1)[0]
    # interval_filter = interval_.abs() > 0.05
    # partial_interval_lengths = interval_lengths[interval_filter]
    # partial_interval_coverages = interval_coverages[interval_filter]

    # For each quantile level, we compute the absolute correlation of the interval length and
    # interval coverage in the batch. Then, we take the mean along the quantile level dimension.
    corr = pearson_corr(interval_lengths, interval_coverages, dim=0).abs().mean()
    return corr
