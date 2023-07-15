import math
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from tueplots import fonts

from .constants import base_model_names, metric_names
from .dataframes import build_grouped_comparison_df
from .stats import cohen_d


def debug_grouped_size(grouped):
    def agg(x):
        if len(x) > 5:
            display(x)

    grouped.apply(agg)


def build_cohen_d(df, metrics, baseline_query, join_by, columns_to_keep):
    grouped = build_grouped_comparison_df(df, metrics, baseline_query, join_by, columns_to_keep)
    sizes = grouped.size().value_counts().to_dict()
    print(
        'Size of groups:',
        ', '.join([f'{count} of size {size}' for size, count in sizes.items()]),
    )
    # debug_grouped_size(grouped)
    for size in sizes:
        assert size <= 10, size
    df_cohen = grouped.apply(
        lambda x: cohen_d(
            x['compared'].astype(float).to_numpy(),
            x['baseline'].astype(float).to_numpy(),
        )
    )
    df_cohen = df_cohen.rename("Cohen's d").reset_index()
    df_cohen['metric'] = pd.Categorical(df_cohen['metric'], metrics)
    df_cohen = df_cohen.sort_values('metric', kind='stable')
    return df_cohen


def symmetrize_x_axis(axis):
    x_lim = np.abs(axis.get_xlim()).max()
    axis.set_xlim(xmin=-x_lim, xmax=x_lim)


def plot_cohen_d_barplot(df, col_queries, legend=True, figsize=None):
    n = len(col_queries)
    nrows, ncols = 1, n
    names = df.name.unique()
    if figsize is None:
        figsize = (5 * ncols, 0.3 * len(names) * nrows)
    fig, axes = plt.subplots(
        1,
        n,
        sharex=False,
        sharey=True,
        squeeze=False,
        figsize=figsize,
        dpi=300,
    )

    for axis, (name, query), i in zip(axes.flatten(), col_queries.items(), range(n)):
        data = df.query(query)
        if data["Cohen's d"].isna().all():
            continue
        model_name = data.apply(lambda d: f'{d["base_loss"]}\n({d["pred_type"]})', axis='columns')
        axis.axvline(x=0, color='black', ls='--', zorder=1)
        g = sns.barplot(
            data,
            x="Cohen's d",
            y=model_name,
            hue='model',
            orient='h',
            errorbar=('se', 1),
            capsize=0.1,
            errwidth=1,
            ax=axis,
        )
        symmetrize_x_axis(axis)
        g.legend_.remove()
        axis.set_xlabel(f"Cohen's d of\n{name}", fontsize=9)
        axis.tick_params(axis='both', which='major', labelsize=9)

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # title='Regularization',
            loc='lower center',
            bbox_to_anchor=(0.5, 1 - 0.05),
            frameon=True,
            ncol=4,
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def plot_sorted_boxplot(df, x=None, y=None, *args, **kwargs):
    medians = df.groupby(y)[x].median().sort_values()
    return sns.boxenplot(df, x=x, y=y, order=medians.index, *args, **kwargs)


def posthoc_or_regul_color_map(df):
    blues = sns.color_palette('Blues', 4)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 4)[1:]

    method_map = {
        'Rec-KDE': blues[0],
        'Rec-LIN': blues[1],
        'CQR': blues[1],
        'Rec-EMP': blues[2],
        'QR': greens[0],
        'Trunc': greens[0],
        'PCE-KDE': greens[1],
        'PCE-Sort': greens[2],
    }
    color_map = {}
    for base_loss in df['base_loss'].unique():
        for name, color in method_map.items():
            color_map[f'{base_model_names[base_loss]} + {name}'] = color
        color_map[f'{base_model_names[base_loss]}'] = reds[1]
    return color_map


def posthoc_dataset_color_map(df):
    blues = sns.color_palette('Blues', 4)[1:]
    greens = sns.color_palette('Greens', 4)[1:]
    reds = sns.color_palette('Reds', 4)[1:]

    names = df.name.unique()
    end = ' (calib)'
    posthoc_names = [name[: -len(end)] for name in names if name.endswith(end)]
    color_map = {}
    for i, posthoc_name in enumerate(posthoc_names):
        color_map[f'{posthoc_name} (train)'] = blues[i]
        color_map[f'{posthoc_name} (calib)'] = greens[i]
    i = 0
    for name in names:
        if name not in color_map:
            color_map[name] = reds[i]
            i += 1
    return color_map


def plot_cohen_d_boxplot(df, col_queries, legend=True, figsize=None, color_map_name=None, ncols=2):
    col_queries = {name: query for name, query in col_queries.items() if len(df.query(query)) > 0}
    size = len(col_queries)
    nrows = math.ceil(size / ncols)
    names = df.name.unique()
    if figsize is None:
        # figsize = (3.5 * ncols, 1. + 0.15 * len(names) * nrows)
        figsize = (4.5 * ncols, 1.6 + 0.15 * len(names) * nrows)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        sharex=False,
        sharey=False,
        squeeze=False,
        figsize=figsize,
        dpi=300,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)

    if color_map_name == 'posthoc_or_regul':
        color_map = posthoc_or_regul_color_map(df)
    elif color_map_name == 'posthoc_dataset':
        color_map = posthoc_dataset_color_map(df)
    else:
        color_map = {name: color for name, color in zip(names, sns.color_palette(n_colors=len(names)))}

    for axis, (name, query) in zip(ax_flatten, col_queries.items()):
        # print(query, len(df.query(query)))
        data = df.query(query)
        if data["Cohen's d"].isna().all():
            continue
        axis.axvline(x=0, color='black', ls='--', zorder=1)
        g = plot_sorted_boxplot(
            data,
            x="Cohen's d",
            y='name',
            orient='h',
            linewidth=1,
            palette=color_map,  # fliersize=2,
            ax=axis,
            flier_kws={'s': 2},
        )
        g.set_xscale('symlog')
        symmetrize_x_axis(axis)
        metric_name = metric_names[name.split('_', 1)[-1]]
        axis.set_xlabel(f"Cohen's d of {metric_name}", fontsize=9)
        axis.set_ylabel('')
        axis.tick_params(axis='both', which='major', labelsize=9)

    if legend:
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            # title='Regularization',
            loc='lower center',
            bbox_to_anchor=(0.5, 1 - 0.05),
            frameon=True,
            ncol=4,
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def series_to_int(series):
    d = {value: order for order, value in enumerate(series.unique())}
    return series.map(d)


def plot_cohen_d_indexed(df):
    df = df.copy()
    df['dataset'] = series_to_int(df['dataset'])
    df = df.sort_values('dataset', kind='stable')
    metrics = df.metric.unique()
    n = len(metrics)
    nrows, ncols = n, 1
    fig, axes = plt.subplots(
        n,
        1,
        sharex=True,
        squeeze=False,
        figsize=(4 * ncols, 2 * nrows),
        dpi=300,
    )
    for axis, (metric, df_metric), i in zip(axes.flatten(), df.groupby('metric'), range(n)):
        g = sns.scatterplot(
            df_metric,
            x='dataset',
            y="Cohen's d",
            hue='model',
            style='pred_type',
            legend='full',
            ax=axis,
        )
        if i != n - 1:
            g.set(xlabel=None)
        axis.set_title(metric)

    fig.tight_layout()
    return fig
