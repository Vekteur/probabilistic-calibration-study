import torch
from torch.distributions import AffineTransform, ComposeTransform, CumulativeDistributionTransform

from uq.models.pred_type.ecdf import EmpiricalCDF, StochasticEmpiricalCDF
from uq.models.pred_type.linear_spline import LinearSpline
from uq.models.pred_type.mixture_dist import LogisticMixtureDist
from uq.models.pred_type.reflected_dist import ReflectedDist
from uq.models.pred_type.truncated_dist import TruncatedDist
from uq.utils.torch_utils import centered_bins

PostHocPitCalibration = EmpiricalCDF
PostHocStochasticPitCalibration = StochasticEmpiricalCDF


class PostHocLinearPitCalibration(LinearSpline):
    def __init__(self, pit, **kwargs):
        bx = torch.sort(pit)[0]
        # centered_bins(len(bx))
        by = (torch.arange(len(bx)) + 1) / (len(bx) + 1)
        super().__init__(bx, by, **kwargs)


class SmoothEmpiricalCDF(CumulativeDistributionTransform):
    def __init__(self, x, s=100.0, model=None, **kwargs):
        self.model = model
        dist = LogisticMixtureDist(x, torch.tensor(1.0 / s), probs=torch.ones_like(x), model=model)
        dist = ReflectedDist(dist, torch.tensor(0.0), torch.tensor(1.0))
        dist = TruncatedDist(dist, torch.tensor(0.0), torch.tensor(1.0))
        super().__init__(dist, **kwargs)

    def _call(self, x):
        y = super()._call(x)
        # self.plot(x, y)
        return y

    def plot(self, x, y):
        import matplotlib.pyplot as plt
        import numpy as np

        from uq.utils.general import print_once, savefig

        print_once('plot', 'Plot enabled during training')

        fig, axis = plt.subplots()
        x, y = x.detach().numpy(), y.detach().numpy()
        idx = np.argsort(x)
        axis.plot(x[idx], y[idx], '-bo')
        axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set(adjustable='box', aspect='equal')
        savefig(f'tmp/posthoc_call/{self.model.current_epoch}')


PostHocSmoothPitCalibration = SmoothEmpiricalCDF


class Adjuster(AffineTransform):
    def _call(self, x):
        return super()._call(x).clamp(0, 1)


class PostHocConformalCalibration:
    def __init__(self, y, quantiles, alpha, scaler=None, rc=None):
        self.q = self.conformal_quantile(y, quantiles, alpha)
        if scaler is not None and rc.config.unnormalize:
            self.q *= scaler.scale_

    def conformal_quantile(self, y, quantiles, alpha):
        N, n_quantiles = quantiles.shape
        scores = y - quantiles
        # We sort along the batch dimension
        scores = torch.sort(scores, dim=0)[0]
        assert scores.shape == (N, n_quantiles)
        indices = torch.ceil((N + 1) * alpha) - 1
        # In the extreme case that an element of `indices` is not a valid index, we clamp it
        indices = torch.clamp(indices, 0, N - 1)
        # torch.quantile does not work if we want different quantiles levels for different indices
        return scores[indices.long(), torch.arange(scores.shape[1])]

    def transform(self, quantiles):
        assert quantiles.shape[1] == self.q.shape[0]
        return quantiles + self.q
