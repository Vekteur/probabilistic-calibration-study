import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tueplots import fonts


def plot_sorted_boxplot(df, metrics):
    df = df[metrics].reset_index(level='run_id', drop=True)
    df = df.groupby(df.index.names, dropna=False).mean().reset_index()

    df = df.reset_index()[metrics + ['name']]
    n = len(metrics)
    nrows, ncols = 1, n
    figsize = (5 * ncols, 4 * nrows)
    fig, axes = plt.subplots(
        1,
        n,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=figsize,
        dpi=300,
    )

    for axis, metric, i in zip(axes.flatten(), metrics, range(n)):
        medians = df.groupby('name')[metric].median().sort_values()
        sns.boxplot(df, x=metric, y='name', orient='h', order=medians.index, ax=axis)
        if metric in ['CRPS', 'WIS', 'MAE', 'RMSE']:
            axis.set_xscale('log')
        elif metric in ['NLL']:
            axis.set_xscale('symlog')
        axis.set(xlabel=metric, ylabel=None)

    fig.tight_layout()
    return fig
