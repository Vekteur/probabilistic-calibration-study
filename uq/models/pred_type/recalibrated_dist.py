import torch
from torch.distributions import Normal, TransformedDistribution, Uniform
from torch.distributions.transforms import (
    AffineTransform,
    ComposeTransform,
    CumulativeDistributionTransform,
)

from uq.utils.dist import adjust_unit_tensor


class UnitUniform(Uniform):
    def __init__(self, batch_shape, *args, **kwargs):
        super().__init__(torch.zeros(batch_shape), torch.ones(batch_shape), *args, **kwargs)

    def log_prob(self, value):
        value = adjust_unit_tensor(value)
        eps = 1e-6
        value[value == 0.0] += eps
        value[value == 1.0] -= eps
        return super().log_prob(value)


class RecalibratedDist(TransformedDistribution):
    def __init__(self, dist, posthoc_model, scaler=None):
        self.dist = dist
        self.posthoc_model = posthoc_model
        base_dist = UnitUniform(dist.batch_shape)

        transforms = [
            self.posthoc_model.inv,
            CumulativeDistributionTransform(dist).inv,
        ]
        if scaler is not None:
            transforms.append(AffineTransform(scaler.mean_, scaler.scale_))

        super().__init__(base_dist, transforms)

    def crps(self, value):
        raise NotImplementedError()

    def unnormalize(self, scaler):
        return RecalibratedDist(self.dist, self.posthoc_model, scaler)
