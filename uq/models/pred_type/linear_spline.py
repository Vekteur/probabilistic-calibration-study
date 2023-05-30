import operator
import warnings

import torch
from torch.distributions import Transform, constraints

from uq.utils.dist import adjust_unit_tensor


def is_sorted(x, strictly=False):
    op = operator.lt if strictly else operator.le
    return op(x[..., :-1], x[..., 1:]).all().item()


def assert_sorted(x, strictly=False):
    op = operator.lt if strictly else operator.le
    sorted_mask = op(x[..., :-1], x[..., 1:])
    not_sorted1 = x[~torch.cat((sorted_mask, torch.tensor([True])))]
    not_sorted2 = x[~torch.cat((torch.tensor([True]), sorted_mask))]
    assert is_sorted(x, strictly=strictly), (not_sorted1, not_sorted2)


def groupby_agg_mean(bx, by):
    bx, inverse_indices, counts = torch.unique_consecutive(bx, return_inverse=True, return_counts=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        by = torch.zeros_like(bx, dtype=torch.float).scatter_add_(0, inverse_indices, by)
        by = by / counts.float()
    return bx, by


def interp1d(inputs, bx, by):
    # searchsorted requires the innermost dimension to be sorted
    bins = torch.searchsorted(bx, inputs, side='right')
    dx = bx.diff()
    assert (dx > 0.0).all()
    assert (0 < bins).all(), inputs[bins == 0]
    assert ((0 < bins) & (bins < bx.shape[0])).all()
    assert not torch.isnan(inputs).any()
    # print(torch.nonzero(torch.isnan(inputs)), flush=True)
    shift = (inputs - bx[bins - 1]) / dx[bins - 1]
    mask = (0.0 <= shift) & (shift <= 1.0)
    assert mask.all(), shift[~mask]
    res = by[bins - 1] * (1 - shift) + by[bins] * shift
    return adjust_unit_tensor(res)


class LinearSpline(Transform):
    domain = constraints.unit_interval
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __init__(self, bx, by, **kwargs):
        # We assume that x is increasing and y is strictly increasing
        # If x values are duplicated, only one value is kept, and the corresponding y
        # is the mean over all y for this x
        super().__init__(**kwargs)
        assert bx.shape == by.shape
        bx, by = adjust_unit_tensor(bx), adjust_unit_tensor(by)
        bx, by = self.add_bounds(bx, by)
        assert bx[0] == 0, bx[0]
        self.bx, self.by = groupby_agg_mean(bx, by)
        # Make sure to keep the points (0, 0) and (1, 1) after aggregation
        self.by[0] = 0
        self.by[-2] = 1
        assert self.bx[0] == 0, self.bx[0]
        assert_sorted(self.bx)
        assert_sorted(self.by, strictly=True)

    def add_bounds(self, bx, by):
        bound_shape = bx.shape[:-1] + (1,)
        bound_left = torch.full(bound_shape, 0.0)
        bound_right = torch.full(bound_shape, 1.0)
        bound_right_right = torch.full(bound_shape, 2.0)   # Necessary to handle inputs=0
        bx = torch.cat((bound_left, bx, bound_right, bound_right_right), dim=-1)
        by = torch.cat((bound_left, by, bound_right, bound_right_right), dim=-1)
        return bx, by

    def _call(self, x):
        x = adjust_unit_tensor(x)
        return interp1d(x, self.bx, self.by)

    def _inverse(self, y):
        y = adjust_unit_tensor(y)
        return interp1d(y, self.by, self.bx)

    def log_abs_det_jacobian(self, x, y):
        x = adjust_unit_tensor(x)
        bins = torch.searchsorted(self.bx, x, side='right')
        d = self.by.diff() / self.bx.diff()
        return d[bins - 1].log()
