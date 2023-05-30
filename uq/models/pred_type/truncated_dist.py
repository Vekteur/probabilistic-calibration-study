import torch
from torch.distributions import Distribution, constraints

from uq.utils.dist import adjust_tensor, adjust_unit_tensor


class TruncatedDist(Distribution):
    support = constraints.unit_interval
    has_rsample = True   # It depends on the parent class but we don't have access to it here

    def __init__(self, dist, a=-torch.inf, b=torch.inf):
        self.dist = dist
        self.a = a
        self.b = b
        self.Fa, self.Fb = self.dist.cdf(torch.tensor([self.a, self.b]))

    def cdf(self, value):
        value = adjust_tensor(value, self.a, self.b)
        res = (self.dist.cdf(value) - self.Fa) / (self.Fb - self.Fa)
        res[value < self.a] = 0
        res[self.b < value] = 1
        return res

    def icdf(self, value):
        value = adjust_unit_tensor(value)
        value = value * (self.Fb - self.Fa) + self.Fa
        return self.dist.icdf(value)

    def log_prob(self, value):
        value = adjust_tensor(value, self.a, self.b)
        res = self.dist.log_prob(value) / (self.Fb - self.Fa)
        res[value < self.a] = -torch.inf
        res[self.b < value] = -torch.inf
        return res

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.dist.batch_shape
        rand = torch.rand(shape, device=self.a.device)
        return self.icdf(rand)
