import torch
from torch.distributions import Transform, constraints

from uq.utils.dist import adjust_unit_tensor


class StochasticEmpiricalCDF(Transform):
    domain = constraints.real
    codomain = constraints.unit_interval
    bijective = True
    sign = +1

    def __init__(self, samples, sep_epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.sep_epsilon = sep_epsilon
        sort_idx = torch.argsort(samples)

        # Group values that are less than sep_epsilon apart
        # Compute the index at the start of each group (except the first group)
        group_idx = (samples[sort_idx].diff() > sep_epsilon).nonzero().int()[:, 0] + 1
        group_size = group_idx.diff(prepend=torch.tensor([0]), append=torch.tensor([len(sort_idx)]))
        groups = torch.split(sort_idx, list(group_size))
        self.grouped_samples = torch.tensor([samples[group].mean() for group in groups])
        self.group_prop = group_size / len(samples)
        self.group_cum_prop = self.group_prop.cumsum(0)
        self.group_cum_prop = torch.cat((torch.tensor([0.0]), self.group_cum_prop))
        self.grouped_samples = torch.cat((self.grouped_samples, torch.tensor([torch.inf])))
        # The value of the cdf at the group of index `idx` is `group_cum_size[idx]` to the left and
        # `group_cum_size[idx + 1]` to the right

    def _call(self, x):
        x = adjust_unit_tensor(x)
        group_idx = torch.searchsorted(self.grouped_samples, x - self.sep_epsilon, side='right')
        y = self.group_cum_prop[group_idx]
        in_group = (self.grouped_samples[group_idx] - x).abs() < self.sep_epsilon
        y[in_group] += self.group_prop[group_idx[in_group]] * torch.rand(group_idx[in_group].shape)
        return y

    def _inverse(self, y):
        group_idx = torch.searchsorted(self.group_cum_prop, y, side='left')
        x = torch.empty_like(y, dtype=y.dtype)
        x[group_idx > 0] = self.grouped_samples[group_idx[group_idx > 0] - 1]
        x[group_idx == 0] = 0   # Special case: when y == 0, we define the corresponding quantile as 0
        return x

    def log_abs_det_jacobian(self, x, y):
        x = adjust_unit_tensor(x)
        group_idx = torch.searchsorted(self.grouped_samples, x - self.sep_epsilon, side='right')
        in_group = (self.grouped_samples[group_idx] - x).abs() < self.sep_epsilon
        return torch.where(in_group, torch.inf, -torch.inf)


class EmpiricalCDF(StochasticEmpiricalCDF):
    def __init__(self, samples):
        super().__init__(samples, sep_epsilon=0)
