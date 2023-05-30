import torch


def unnormalize_y(y, scaler, rc):
    if rc.config.unnormalize:
        y = y.unsqueeze(dim=-1)
        y = scaler.inverse_transform(y)
        y = y.squeeze(dim=-1)
    return y


def unnormalize_dist_y(dist, y, scaler, rc):
    if rc.config.unnormalize:
        dist = dist.unnormalize(scaler)
        y = unnormalize_y(y, scaler, rc)
    return dist, y


def unnormalize_quantiles(quantiles, y, scaler, rc):
    if rc.config.unnormalize:
        # We should use unnormalize_y for quantiles too because it also needs to be unsequeezed
        quantiles = unnormalize_y(quantiles, scaler, rc)
        y = unnormalize_y(y, scaler, rc)
    return quantiles, y


def adjust_tensor(x, a=0.0, b=1.0, *, epsilon=1e-4):
    # We accept that, due to rounding errors, x is not in the interval up to epsilon
    mask = (a - epsilon <= x) & (x <= b + epsilon)
    assert mask.all(), (x[~mask], a, b)
    return x.clamp(a, b)


def adjust_unit_tensor(x, epsilon=1e-4):
    return adjust_tensor(x, a=0.0, b=1.0, epsilon=epsilon)


def icdf_from_cdf(dist, alpha, epsilon=1e-5, warn_precision=4e-3, low=None, high=None):
    """
    Compute the quantiles of a distribution using binary search, in a vectorized way.
    """

    # print('alpha:', alpha.shape, dist.batch_shape, flush=True)
    alpha = adjust_unit_tensor(alpha)
    alpha, _ = torch.broadcast_tensors(alpha, torch.zeros(dist.batch_shape))
    # n_levels, batch_size = alpha.shape
    # assert dist.batch_shape == (batch_size,)
    # try:
    #     torch.broadcast_tensors(dist.cdf(low), alpha)
    # except:
    #     print(dist.cdf(low).shape, alpha.shape)
    #     raise
    # Expand to the left and right until we are sure that the quantile is in the interval
    expansion_factor = 4
    if low is None:
        low = torch.full(alpha.shape, -1.0)
        while (mask := dist.cdf(low) > alpha + epsilon).any():
            low[mask] *= expansion_factor
    else:
        low = low.clone()
    if high is None:
        high = torch.full(alpha.shape, 1.0)
        while (mask := dist.cdf(high) < alpha - epsilon).any():
            high[mask] *= expansion_factor
    else:
        high = high.clone()
    low, high, _ = torch.broadcast_tensors(low, high, torch.zeros(alpha.shape))
    assert dist.cdf(low).shape == alpha.shape

    # Binary search
    prev_precision = None
    while True:
        # To avoid "UserWarning: Use of index_put_ on expanded tensors is deprecated".
        low = low.clone()
        high = high.clone()
        precision = (high - low).max()
        # Stop if we have enough precision
        if precision < 1e-5:
            break
        # Stop if we can not improve the precision anymore
        if prev_precision is not None and precision >= prev_precision:
            break
        mid = (low + high) / 2
        mask = dist.cdf(mid) < alpha
        low[mask] = mid[mask]
        high[~mask] = mid[~mask]
        prev_precision = precision

    if precision > warn_precision:
        pass
        # log.warn(f'Imprecise quantile computation with precision {precision}')
    return low


def icdf(dist, levels):
    try:
        return dist.icdf(levels)
    except NotImplementedError:
        return icdf_from_cdf(dist, levels)
