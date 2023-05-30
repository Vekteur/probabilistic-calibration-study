import torch
from pyro.distributions.transforms import AffineTransform, ComposeTransform, TanhTransform
from torch.distributions import constraints
from torch.distributions.distribution import Distribution

from uq.models.pred_type.rational_spline import RationalSpline


class SplineDist(Distribution):
    arg_constraints = {
        'w': constraints.real,
        'h': constraints.real,
        'd': constraints.real,
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, w, h, d, count_bins=8, scaler=None, validate_args=None):
        self.w, self.h, self.d = w, h, d
        self.count_bins = count_bins
        self.transform = self.make_transform(scaler=scaler)
        batch_shape = self.w.shape[:-1]
        super().__init__(batch_shape, validate_args=validate_args)

    def make_transform(self, scaler=None):
        transforms = [
            # x is in [0, 1]
            AffineTransform(-1, 2),
            # x is in [-1, 1]
            RationalSpline(
                (self.w, self.h, self.d),
                1,
                count_bins=self.count_bins,
                bound=1.0,
                order='quadratic',
            ),
            # x is in [-1, 1]
            TanhTransform().inv,
            # x is in R
        ]
        if scaler is not None:
            # Normalization: X = (Y - a) / b = -a / b + Y / b
            # Unnormalization: Y = a + bX
            unnormalization_transform = AffineTransform(scaler.mean_, scaler.scale_)
            transforms.append(unnormalization_transform)
        transform = ComposeTransform(transforms)
        return transform

    def unnormalize(self, scaler):
        return SplineDist(self.w, self.h, self.d, scaler=scaler)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SplineDist, _instance)
        batch_shape = torch.Size(batch_shape)
        new.w = self.w.expand(batch_shape)
        new.h = self.h.expand(batch_shape)
        new.d = self.d.expand(batch_shape)
        super(SplineDist, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        new.make_transform()
        return new

    def cdf(self, value):
        value, _ = torch.broadcast_tensors(value, torch.zeros(self.batch_shape))
        return self.transform.inv(value)

    def icdf(self, value):
        value, _ = torch.broadcast_tensors(value, torch.zeros(self.batch_shape))
        return self.transform(value)

    def log_prob(self, value):
        value, _ = torch.broadcast_tensors(value, torch.zeros(self.batch_shape))
        output = self.cdf(value)
        return self.transform.inv.log_abs_det_jacobian(value, output)

    def rsample(self, sample_shape=torch.Size()):
        # shape = self._extended_shape(sample_shape)
        shape = torch.Size(sample_shape) + self.batch_shape
        rand = torch.rand(shape, dtype=self.w.dtype, device=self.w.device)
        return self.transform(rand)

    def crps(self, value):
        raise NotImplementedError()
