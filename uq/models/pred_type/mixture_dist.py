import torch
import torch.nn.functional as F
from pyro.distributions import Logistic
from torch.distributions import Categorical, MixtureSameFamily, Normal

from uq.metrics.quantiles import crps_normal_mixture
from uq.utils.dist import icdf_from_cdf


# Mixture distribution for a location-scale family
class MixtureDist(MixtureSameFamily):
    def __init__(self, component_dist_class, means, stds, *, probs=None, logits=None):
        mix_dist = Categorical(probs=probs, logits=logits)
        self.component_dist_class = component_dist_class
        component_dist = self.component_dist_class(means, stds)
        super().__init__(mix_dist, component_dist)

    def icdf(self, value):
        return icdf_from_cdf(self, value)

    def affine_transform(self, loc, scale):
        """
        Let $X ~ Dist(\mu, \sigma)$. Then $a + bX ~ Dist(a + b \mu, b \sigma)$.
        The reasoning is similar for a mixture.
        """
        component_dist = self.component_distribution
        mix_dist = self.mixture_distribution
        means, stds = component_dist.loc, component_dist.scale
        means = loc + scale * means
        stds = scale * stds
        return type(self)(means, stds, logits=mix_dist.logits)

    def unnormalize(self, scaler):
        return self.affine_transform(scaler.mean_, scaler.scale_)

    def normalize(self, scaler):
        return self.affine_transform(-scaler.mean_ / scaler.scale_, 1.0 / scaler.scale_)

    def rsample(self, sample_shape=torch.Size(), tau=1):
        """
        Args:
            n_samples (int, optional): Number of samples.
            tau (int, optional): Argument to `gumbel_softmax`.

        Returns:
            Tensor: A tensor of shape `[batch_size, n_samples]`
        """

        logits = self.mixture_distribution.logits
        categ_one_hot_samples = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        loc = self.component_distribution.loc[categ_one_hot_samples == 1]
        scale = self.component_distribution.scale[categ_one_hot_samples == 1]
        return self.component_dist_class(loc, scale).rsample(sample_shape)


class NormalMixtureDist(MixtureDist):
    def __init__(self, *args, **kwargs):
        super().__init__(Normal, *args, **kwargs)

    def crps(self, value):
        return crps_normal_mixture(self, value)


class LogisticMixtureDist(MixtureDist):
    def __init__(self, *args, model=None, **kwargs):
        super().__init__(Logistic, *args, **kwargs)
        self.model = model
        # self.plot()

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        from uq.utils.general import print_once, savefig

        print_once('plot', 'Plot enabled during training')

        fig, axis = plt.subplots()
        x = torch.linspace(0, 1, 100)
        with torch.no_grad():
            y = self.cdf(x)
        axis.plot(x.numpy(), y.numpy(), '-bo')
        axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.set(adjustable='box', aspect='equal')
        savefig(f'tmp/rec-kde/{self.model.current_epoch}')
