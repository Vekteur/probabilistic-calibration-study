import numpy as np


def cohen_d(x1, x2):
    # By default numpy uses ddof=0, i.e., var is biased
    v1 = x1.var(axis=-1)
    v2 = x2.var(axis=-1)
    s = np.sqrt((v1 + v2) / (x1.shape[-1] + x2.shape[-1] - 2))
    return (x1.mean(axis=-1) - x2.mean(axis=-1)) / s
