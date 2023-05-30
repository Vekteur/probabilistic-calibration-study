import torch
import torch.nn.functional as F
from torch.distributions import Normal

from uq.utils.dist import icdf_from_cdf

from .mixture_dist import NormalMixtureDist


class NormalDist(Normal):
    def __init__(self, mean, std):
        super().__init__(mean, std)

    def to_mixture(self):
        return NormalMixtureDist(self.loc, self.scale, probs=torch.tensor(1.0))

    def crps(self, value):
        return self.to_mixture().crps(value)

    def int_power_normal_pdf(self, p):
        # https://math.stackexchange.com/questions/2357263/integral-of-a-power-of-a-normal-distribution
        return ((2 * torch.tensor(torch.pi)).sqrt() * self.scale) ** (1 - p) / torch.sqrt(torch.tensor(p))

    def qs(self, value):
        return -2 * self.log_prob(value).exp() + self.int_power_normal_pdf(2)

    def affine_transform(self, loc, scale):
        mean = self.loc + scale * self.mean
        std = scale * std
        return NormalDist(mean, std)

    def unnormalize(self, scaler):
        return self.affine_transform(scaler.mean_, scaler.scale_)

    def normalize(self, scaler):
        return self.affine_transform(-scaler.mean_ / scaler.scale_, 1.0 / scaler.scale_)
