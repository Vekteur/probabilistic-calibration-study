import torch

from .base_dist_module import DistModule
from uq.models.pred_type.mixture_dist import NormalMixtureDist
from uq.models.regul.oqr_regul import oqr_loss


class DistOQR_Regul(DistModule):
    def __init__(self, *args, s=100., **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
    
    def compute_regul(self, dist, y):
        return dist_oqr_loss(dist, y, s=self.hparams.s)


def sample(dist, n_samples, tau):
    sample_kwargs = {}
    if type(dist) == NormalMixtureDist:
        sample_kwargs['tau'] = tau
    samples = dist.rsample(sample_shape=[n_samples], **sample_kwargs).swapaxes(0, 1)
    sorted_samples = torch.sort(samples, dim=1)[0]
    return sorted_samples


def dist_oqr_loss(dist, y, n_samples=100, tau=1, s=100):
    sorted_samples = sample(dist, n_samples, tau)
    return oqr_loss(sorted_samples, y, s=s)
