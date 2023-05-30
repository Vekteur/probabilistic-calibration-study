"""
Adapted from https://docs.pyro.ai/en/stable/_modules/pyro/distributions/transforms/spline.html.
"""

import torch
import torch.nn.functional as F
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions.transforms.spline import ConditionedSpline
from pyro.distributions.util import copy_docs_from
from torch.distributions import constraints


@copy_docs_from(ConditionedSpline)
class RationalSpline(ConditionedSpline, TransformModule):

    domain = constraints.real
    codomain = constraints.real
    bijective = True
    sign = +1

    def __init__(self, params, input_dim, count_bins=8, bound=3.0, order='linear'):
        super(RationalSpline, self).__init__(self._params)

        self.input_dim = input_dim
        self.count_bins = count_bins
        self.bound = bound
        self.order = order

        if self.order == 'linear':
            self.w, self.h, self.d, self.l = params
        elif self.order == 'quadratic':
            self.w, self.h, self.d = params
        else:
            raise ValueError(
                "Keyword argument 'order' must be one of ['linear', 'quadratic'], but '{}' was found!".format(
                    self.order
                )
            )

    def _params(self):
        # widths, unnormalized_widths ~ (input_dim, num_bins)
        w = F.softmax(self.w, dim=-1)
        h = F.softmax(self.h, dim=-1)
        d = F.softplus(self.d)
        if self.order == 'linear':
            l = torch.sigmoid(self.l)
        else:
            l = None
        return w, h, d, l
