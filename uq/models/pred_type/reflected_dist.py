import torch
from torch.distributions import Distribution, constraints

from uq.utils.dist import adjust_tensor, adjust_unit_tensor, icdf_from_cdf


def logsumexp(a, dim=-1, b=None, keepdims=False):
    if b is not None:
        a, b = torch.broadcast_tensors(a, b)
        if (b == 0).any():
            a = a + 0.0  # promote to at least float
            a[b == 0] = -torch.inf

    a_max = torch.amax(a, dim=dim, keepdims=True)

    if a_max.ndim > 0:
        a_max[~torch.isfinite(a_max)] = 0
    elif not torch.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = torch.as_tensor(b)
        tmp = b * torch.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    s = torch.sum(tmp, dim=dim, keepdims=keepdims)
    out = torch.log(s)

    if not keepdims:
        a_max = torch.squeeze(a_max, dim=dim)
    out += a_max
    return out


class ReflectedDist(Distribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, dist, a=-torch.inf, b=torch.inf):
        self.dist = dist
        self.a = a
        self.b = b

    @property
    def batch_shape(self):
        return self.dist._batch_shape

    def cdf(self, value):
        value = adjust_tensor(value, self.a, self.b)
        res = 1 - self.dist.cdf(2 * self.b - value) + self.dist.cdf(value) - self.dist.cdf(2 * self.a - value)
        res[value < self.a] = 0
        res[self.b < value] = 1
        return res

    def icdf(self, value):
        value = adjust_unit_tensor(value)
        return icdf_from_cdf(self, value, low=self.a, high=self.b)

    def log_prob(self, value):
        value = adjust_tensor(value, self.a, self.b)
        # log_probs = torch.stack([
        #     self.dist.log_prob(2 * self.b - value),
        #     self.dist.log_prob(value),
        #     self.dist.log_prob(2 * self.a - value)
        # ], dim=-1)

        # res = logsumexp(log_probs, dim=-1)

        res = (
            self.dist.log_prob(2 * self.b - value).exp()
            + self.dist.log_prob(value).exp()
            + self.dist.log_prob(2 * self.a - value).exp()
        ).log()

        res[value < self.a] = -torch.inf
        res[self.b < value] = -torch.inf
        return res

    def sample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.dist.batch_shape
        rand = torch.rand(shape, device=self.a.device)
        return self.icdf(rand)
