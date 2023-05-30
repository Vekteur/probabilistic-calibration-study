import math

import torch

from uq.utils.fast_soft_sort.pytorch_ops import soft_sort
from uq.utils.torch_utils import centered_bins

from .indicator import indicator

"""
The `compute_neural_sort` function is adapted from https://github.com/ermongroup/neuralsort.
The `qr_loss` and `neg_entropy` functions are adapted from https://github.com/occam-ra-zor/QR.
"""


def compute_neural_sort(scores, tau=0.001):
    scores = scores.unsqueeze(dim=0).unsqueeze(dim=-1)  # [1,N,1]
    bsize = scores.size()[0]
    dim = scores.size()[1]
    # one = torch.cuda.DoubleTensor(dim, 1).fill_(1)
    one = torch.ones((dim, 1), dtype=torch.float32)

    A_scores = torch.abs(scores - scores.permute(0, 2, 1))
    B = torch.matmul(A_scores, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = (dim + 1 - 2 * (torch.arange(dim) + 1)).float()  # .type(torch.cuda.DoubleTensor)
    C = torch.matmul(scores, scaling.unsqueeze(0))

    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)

    P_hat = sm(P_max / tau)
    return (P_hat @ scores).squeeze(dim=0).squeeze(dim=-1)


def compute_neg_entropy(sorted_pit, spacing=64):
    """
    Compute negative entropy using sample spacing entropy estimation.
    """
    N = sorted_pit.shape[0]
    m = spacing
    m_spacing = sorted_pit[:-m] - sorted_pit[m:]
    # Adding this small quantity avoids rare crashes.
    m_spacing = m_spacing + 1e-5
    # print('m_spacing:', m_spacing.min(), m_spacing.max())
    scaled = -torch.log(m_spacing * ((N + 1) / m))
    neg_entropy = scaled.mean()
    return neg_entropy


def qr_loss(dist, y, spacing=64, neural_sort=False, s=0.001):
    # Change the spacing for extreme batch sizes
    N = y.shape[0]
    if N // 2 == 0:
        return torch.tensor(0)
    elif spacing > N // 2:
        spacing = int(math.sqrt(N))

    pit = dist.cdf(y)
    if neural_sort:
        sorted_pit = compute_neural_sort(pit, tau=s)
    else:
        sorted_pit = torch.sort(pit)[0].flip(0)
    # print(sorted_pit[:5])
    neg_entropy = compute_neg_entropy(sorted_pit, spacing)
    # print(neg_entropy)
    return neg_entropy


def quantile_based_regul(pit, L, neural_sort=False, s=0.1):
    if neural_sort:
        # soft_sort requires a 2D arrays and sorts along the second axis
        sorted_pit = soft_sort(pit[None, :], regularization_strength=s)[0]
    else:
        sorted_pit = torch.sort(pit, dim=-1)[0]
    n = pit.shape[-1]
    lin = (torch.arange(n) + 1) / (n + 1)
    loss = (sorted_pit - lin).abs().pow(L).mean(dim=-1)
    return loss


def cdf_based_regul_from_pit(pit, L, alpha=100, s=None):
    assert pit.dim() == 1, pit.shape
    # alpha can be given as an int indicating the number of levels
    # or directly as a Tensor with the levels
    if type(alpha) == int:
        alpha = centered_bins(alpha)
    # Compute a relaxed expression of indicator(pit <= alpha)
    p = indicator(pit[:, None], alpha, s=s).mean(dim=0)
    assert p.shape == alpha.shape
    loss = (p - alpha).abs().pow(L).mean()
    return loss


def observed_prob(y, quantiles, **kwargs):
    return indicator(y, quantiles, **kwargs).mean(dim=0)


def cdf_based_regul_from_quantiles(quantiles, y, L, alpha, s=None):
    N, n_quantiles = quantiles.shape
    assert y.shape == (N,) and alpha.shape == (n_quantiles,)
    # Compute a relaxed expression of indicator(y <= quantiles)
    p = observed_prob(y[:, None], quantiles, s=s)
    assert p.shape == alpha.shape
    loss = (p - alpha).abs().pow(L).mean()
    return loss


def truncated_dist_regul(quantiles, y, alpha):
    """From the paper "Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification" """
    N, n_quantiles = quantiles.shape
    y = y.reshape(-1, 1)
    p = observed_prob(y.reshape(-1, 1), quantiles)
    assert p.shape == alpha.shape
    case1 = indicator(p, alpha) * ((y - quantiles) * indicator(quantiles, y))
    case2 = indicator(alpha, p) * ((quantiles - y) * indicator(y, quantiles))
    return (case1 + case2).mean()


def qd_regul(quantiles, y, alpha, s=None):
    """From the paper "High-Quality Prediction Intervals for Deep Learning: A Distribution-Free, Ensembled Approach" """
    """ 
    The original loss target the coverage of intervals with coverage 1 - alpha that are not especially centered.
    We apply this loss separately to each quantile by considering the interval [alpha, 1] (which is also of size 1 - alpha)
    """
    N, n_quantiles = quantiles.shape
    assert y.shape == (N,) and alpha.shape == (n_quantiles,)
    p = observed_prob(
        quantiles, y.reshape(-1, 1), s=s
    )   # We use the other sense of the inequality because we at the interval [alpha, 1]
    assert p.shape == alpha.shape
    # We don't multiply by N in this case in order to maintain a similar scale between regularizations
    loss = 1 / (alpha * (1 - alpha)) * torch.max(1 - alpha - p, torch.tensor(0.0)).square()
    loss = loss.mean()
    return loss
