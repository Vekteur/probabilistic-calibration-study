import numpy as np
import torch

from ..base_datamodule import BaseDataModule


def toy_cond_dist(size):
    x = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(10.0)).sample(torch.tensor([size]))
    cond_mean = 2 * (x / 10 + torch.sin(4 * x / 10) + torch.sin(13 * x / 10))
    cond_std = torch.sqrt((x + 1) / 10)
    cond_dist = torch.distributions.Normal(cond_mean, cond_std)
    return x, cond_dist


def generate_data(size):
    x, cond_dist = toy_cond_dist(size)
    return x, cond_dist.sample()


class ToyDataModule(BaseDataModule):
    def use_known_uncertainty(self):
        return False

    def get_data(self):
        prefix, size = self.hparams.name.split('_')
        assert prefix == 'toy'
        x, y = generate_data(int(size))
        return x.numpy().reshape(-1, 1), y.numpy().reshape(-1, 1)
