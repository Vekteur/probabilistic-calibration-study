import torch


def centered_bins(n):
    return (torch.arange(n) + 1.0 / 2) / n
