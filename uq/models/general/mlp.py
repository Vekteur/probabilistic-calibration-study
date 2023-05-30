import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from uq.models.pred_type.mixture_dist import NormalMixtureDist
from uq.models.pred_type.spline_dist import SplineDist


class MLP(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_sizes=[128],
        output_sizes=[1],
        drop_prob=0.2,
        persistent_input_size=0,
        linear_layer=nn.Linear,
    ):

        super().__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.persistent_input_size = persistent_input_size

        self.hidden_layers = nn.ModuleList()
        current_input_size = input_size + persistent_input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(linear_layer(current_input_size, hidden_size))
            current_input_size = hidden_size + persistent_input_size

        self.dropout_layer = nn.Dropout(p=drop_prob)
        self.output_layer = linear_layer(current_input_size, sum(output_sizes))

    def forward(self, x, r=None):
        if self.persistent_input_size > 0:
            if r is None:
                r = torch.rand((x.shape[0], self.persistent_input_size), device=x.device)
            elif r.dim() == 1:
                r = r.unsqueeze(0).repeat(x.shape[0], 1)
        for layer in self.hidden_layers:
            if self.persistent_input_size > 0:
                x = torch.cat([x, r], dim=1)
            x = layer(x)
            x = F.relu(x)
        x = self.dropout_layer(x)
        if self.persistent_input_size > 0:
            x = torch.cat([x, r], dim=1)
        x = self.output_layer(x)
        x = torch.split(x, self.output_sizes, dim=-1)
        return x


class MLP_MixturePrediction(nn.Module):
    def __init__(self, event_size=1, mixture_size=1, add_std_output=True, homoscedastic=False, **kwargs):
        super().__init__()
        self.event_size = event_size
        self.mixture_size = mixture_size
        self.add_std_output = add_std_output
        self.homoscedastic = homoscedastic

        output_sizes = [event_size * mixture_size]
        if self.add_std_output:
            if self.homoscedastic:
                self.rhos = Parameter((torch.rand(event_size * mixture_size) + 1.0) / 10.0)
            else:
                output_sizes.append(event_size * mixture_size)   # rhos
            output_sizes.append(event_size * mixture_size)   # mix_logits
        self.body = MLP(output_sizes=output_sizes, **kwargs)

    def forward(self, x, r=None):
        if self.add_std_output:
            if self.homoscedastic:
                means, mix_logits = self.body(x, r=r)
                rhos = self.rhos
            else:
                means, rhos, mix_logits = self.body(x, r=r)
            stds = F.softplus(rhos) + 0.01
            return means, stds, mix_logits
        else:
            means = self.body(x, r=r)
            return means

    def dist(self, x, r=None):
        assert self.add_std_output
        means, stds, mix_logits = self.forward(x, r=r)
        return NormalMixtureDist(means, stds, logits=mix_logits)


class MLP_SplinePrediction(nn.Module):
    def __init__(self, event_size=1, count_bins=6, **kwargs):
        super().__init__()
        self.event_size = event_size
        self.count_bins = count_bins

        output_sizes = [
            event_size * count_bins,
            event_size * count_bins,
            event_size * (count_bins - 1),
        ]
        self.body = MLP(output_sizes=output_sizes, **kwargs)

    def forward(self, x, r=None):
        w, h, d = self.body(x, r=r)
        return w, h, d

    def dist(self, x, r=None):
        w, h, d = self.forward(x, r=r)
        return SplineDist(w, h, d)


class MLP_QuantilePrediction(nn.Module):
    def __init__(self, event_size=1, n_quantiles=1, monotonic=False, **kwargs):
        super().__init__()
        self.event_size = event_size
        self.n_quantiles = n_quantiles
        self.monotonic = monotonic

        linear_layer = MonotonicLinear if self.monotonic else nn.Linear
        output_size = event_size * n_quantiles
        self.body = MLP(
            output_sizes=[output_size], persistent_input_size=output_size, linear_layer=linear_layer, **kwargs
        )

    def forward(self, x, alpha):
        output = self.body(x, alpha)[0]
        return torch.sort(output, dim=-1)[0]

    def quantiles(self, x, alpha):
        assert (alpha[..., :-1] < alpha[..., 1:]).all()
        n_bins = math.ceil(alpha.shape[0] / self.n_quantiles)
        alpha_bins = alpha.tensor_split(n_bins, dim=0)
        q_list = []
        for alpha_bin in alpha_bins:
            bin_size = alpha_bin.shape[0]
            if bin_size < self.n_quantiles:
                pad_zeros = torch.zeros(self.n_quantiles - bin_size)
                alpha_bin = torch.cat([pad_zeros, alpha_bin], dim=0)
            q_bin = self(x, alpha_bin)
            q_list.append(q_bin[..., :bin_size])
        return torch.cat(q_list, dim=1)


class MonotonicLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, F.softplus(self.weight), self.bias)


class MLP_QuantilePredictionIndependent(nn.Module):
    def __init__(self, hidden_sizes=None, monotonic=False, **kwargs):
        super().__init__()

        self.monotonic = monotonic
        self.last_hidden_size = hidden_sizes[-1]
        linear_layer = MonotonicLinear if self.monotonic else nn.Linear
        self.body = MLP(
            output_sizes=[self.last_hidden_size],
            hidden_sizes=hidden_sizes[:-1],
            linear_layer=linear_layer,
            **kwargs
        )
        self.alpha_layer = linear_layer(1, self.last_hidden_size)
        self.output_layer = linear_layer(self.last_hidden_size, 1)

    def forward(self, x, alpha):
        batch_size = x.shape[0]
        (nb_quantiles,) = alpha.shape
        z1 = self.body(x)[0]
        z2 = self.alpha_layer(alpha[:, None])
        z = z1[:, None, :] + z2[None, :, :]
        assert z.shape == (
            batch_size,
            nb_quantiles,
            self.last_hidden_size,
        ), z.shape
        res = self.output_layer(z)[:, :, 0]
        return res

    def quantiles(self, x, alpha):
        return self(x, alpha)
