import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from uq.analysis.dataframes import get_datasets_df

from .plot_metrics import plot_sem, sem


def plot_all_runs(axis, df, label=None, color=None):
    for run_id, df_run in df.groupby('run_id'):
        label = label if run_id == 0 else None
        axis.plot(df_run['alpha'], df_run['value'], label=label, color=color, lw=0.5)


def plot_agg_runs(axis, df, label=None, color=None):
    df = df.drop(columns='run_id').groupby('alpha')['value'].agg(['mean', sem])
    plot_sem(axis, df, label=label, color=color)


def plot_runs(*args, agg_run=True, **kwargs):
    if agg_run:
        plot_agg_runs(*args, **kwargs)
    else:
        plot_all_runs(*args, **kwargs)


def make_reliability_df(test_df):
    metrics_start = 'test_observed_frequency_'
    metrics = [metric for metric in test_df.columns if metric.startswith(metrics_start)]
    df = (
        test_df[metrics].stack().rename_axis(index={None: 'alpha'}).rename('value').reset_index(level='alpha')
    )
    df['value'] = df['value'].astype(float)
    df['alpha'] = df['alpha'].apply(lambda x: float(x.split('_')[-1]))
    df = df.reset_index()
    df = df.sort_values('alpha', kind='stable')
    return df


def plot_consistency_bands(axis, n, p, coverage):
    low = (1 - coverage) / 2
    high = 1 - low
    assert p.ndim == 1
    low_band, high_band = stats.binom(n, p).ppf(np.array([low, high])[..., None]) / n
    axis.fill_between(p, low_band, high_band, alpha=0.1, color='orange')


def plot_consistency_bands_from_name(axis, ds_name, config, coverage=0.9):
    p = np.linspace(0, 1, 1000)[1:-1]
    ds_df = get_datasets_df(config)
    n = ds_df.query('Dataset == @ds_name')['Total instances'].iloc[0]
    n *= config.dataset_groups.uci.datamodule.args.train_inter_val_calib_test_split_ratio[-1]
    n = int(n)
    plot_consistency_bands(axis, n, p, coverage)


def plot_reliability_diagram(axis, df, agg_run=True, **kwargs):
    plot_runs(axis, df, agg_run=agg_run, **kwargs)
    axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
    axis.set(xlim=(0, 1), ylim=(0, 1))
    axis.set(adjustable='box', aspect='equal')
    # axis.set_aspect('equal', adjustable='box')


def plot_reliability_diagrams(df, agg_run=True, ncols=5, ncols_legend=3):
    datasets = df['dataset'].unique()
    models = df['name'].unique()

    from itertools import cycle

    colors = cycle(mpl.colors.TABLEAU_COLORS)
    colors_dict = dict(zip(models, colors))

    size = len(datasets)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 2, nrows * 2),
        squeeze=False,
        sharex=True,
        sharey=True,
        dpi=200,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)

    for axis, (dataset, df_dataset) in zip(ax_flatten, df.groupby('dataset')):
        for model, df_model in df_dataset.groupby('name'):
            plot_runs(
                axis,
                df_model,
                agg_run=agg_run,
                label=model,
                color=colors_dict[model],
            )
        axis.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1)
        title = dataset
        axis.set(title=title)
        axis.set(xlim=(0, 1), ylim=(0, 1))
        axis.tick_params(axis='both', which='major', labelsize=8)
        axis.tick_params(axis='both', which='minor', labelsize=6)

    for i in range(nrows):
        ax[i, 0].set(ylabel='Observed frequency')
    for i in range(ncols):
        ax[-1, i].set(xlabel='Forecasted probability')

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title='Models',
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        ncol=ncols_legend,
        fontsize=14,
        title_fontsize=14,
    )
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    return fig
