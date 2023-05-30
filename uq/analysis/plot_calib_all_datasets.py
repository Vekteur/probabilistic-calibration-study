import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from tqdm import tqdm

from uq.metrics.calibration import quantile_calibration_from_pits_with_sorting
from uq.utils.general import plot_or_savefig

from .dataframes import get_datasets_df, make_df_abb
from .plot_reliability_diagrams import (
    make_reliability_df,
    plot_consistency_bands_from_name,
    plot_reliability_diagram,
)


def compute_null_hyp(nb_test_samples, n_runs, n):
    pits = np.random.rand(nb_test_samples, n_runs, n)
    calib = quantile_calibration_from_pits_with_sorting(torch.from_numpy(pits), L=1).numpy()
    assert calib.ndim == 2
    return calib.mean(axis=-1)


def hyp_test_uqce(df, test_statistics):
    p_values = {}
    for dataset, df_by_ds in df.groupby('dataset'):
        null_hyp = test_statistics[dataset]
        alternative_hyp = df_by_ds['value'].to_numpy().mean()
        p_value = (null_hyp > alternative_hyp).mean()
        p_values[dataset] = p_value
    return p_values


def compute_barplot_order(plot_df, query, metric):
    return (
        plot_df.query(query)
        .groupby(['dataset_group', 'dataset'])[metric]
        .mean()
        .sort_values()
        .reset_index()
        .dataset
    )


# Adapted from https://github.com/karakatic/statutils/blob/master/statutils/multi_comparison.py
def holm_correction(p):
    n = len(p)
    p0 = np.array(p)
    p_a = p0[np.logical_not(np.isnan(p0))]
    lp = len(p_a)
    i = np.arange(lp)
    o = p_a.argsort()
    ro = np.argsort(o)
    results = np.minimum(1, np.maximum.accumulate((n - i) * p_a[o]))[ro]
    return results


def plot_calib_all_datasets(plot_df, config, order, names=None, test_statistics=None, path=None, print_hyp_test=True):
    metrics = ['test_calib_l1'] + [metric for metric in plot_df.columns if '_observed_frequency_' in metric]
    plot_df = plot_df[metrics].reset_index()

    assert names is not None and pd.Series(names).isin(plot_df.name).all(), plot_df.name.unique()
    plot_df = plot_df.query('name in @names')
    plot_df['name'] = pd.Categorical(plot_df['name'], names)
    plot_df = plot_df.set_index([col for col in plot_df.columns if col not in metrics])

    index = plot_df.index.names
    plot_df = plot_df.reset_index()
    plot_df['dataset'] = pd.Categorical(plot_df['dataset'], order.to_numpy())
    plot_df = plot_df.sort_values('dataset')
    df_abb = make_df_abb(plot_df['dataset'].unique().astype('string'))
    plot_df = plot_df.merge(df_abb)
    plot_df['dataset_idx'] = plot_df['dataset'].factorize()[0]
    plot_df = plot_df.set_index(index)
    plot_df = plot_df.set_index(['dataset_idx', 'abb'], append=True)

    n_rel_diags = 5
    rel_diags_letters = [chr(ord('B') + i) for i in range(n_rel_diags)]
    top_row_mosaic, bottom_row_mosaic = 'A' * n_rel_diags, ''.join(rel_diags_letters)
    fig, ax = plt.subplot_mosaic(f'{top_row_mosaic};{bottom_row_mosaic}', figsize=(9, 4), dpi=300)

    stacked_df = (
        plot_df[metrics].stack().rename_axis(index={None: 'metric'}).to_frame(name='value').reset_index()
    )
    stacked_df['abb'] = stacked_df['abb'].astype('string')

    stacked_df = stacked_df.query(f'metric == "{metrics[0]}"')
    if print_hyp_test:
        p_values_dict = hyp_test_uqce(stacked_df, test_statistics)
        p_values = np.array(list(p_values_dict.values()))
        significance_level = 0.01
        print(
            'Number of significant tests with Bonferroni correction:',
            (np.minimum(p_values * len(p_values), 1) < significance_level).sum(),
        )
        print(
            'Number of significant tests with Holm correction:',
            (holm_correction(p_values) < significance_level).sum(),
        )
    hue = 'name'

    df_calib = pd.DataFrame(test_statistics)
    df_calib.index = df_calib.index.rename('run_id')
    df_calib = df_calib.rename_axis('dataset', axis=1).stack().rename('value').reset_index()
    #df_calib['metric'] = 'test_calib_l1'
    df_calib['name'] = 'Perfectly calibrated'
    df_calib = df_calib.merge(df_abb.reset_index(), on='dataset')
    df_calib['dataset'] = pd.Categorical(df_calib['dataset'], order.to_numpy())
    df_calib = df_calib.sort_values('dataset')
    palette = sns.color_palette('flare')
    palette = sns.color_palette(['orange'])

    orange, blue = sns.color_palette()[1], sns.color_palette()[0]

    g = sns.barplot(
        stacked_df,
        x='abb',
        y='value',
        hue=hue,
        orient='v',
        errorbar=('se', 1),
        capsize=0.1,
        errwidth=1,
        ax=ax['A'],
        #color='orange',
        palette=sns.color_palette([blue]),
    )   
    sns.barplot(
        df_calib,
        x='abb',
        y='value',
        hue=hue,
        orient='v',
        errorbar=None,
        capsize=0.1,
        errwidth=1,
        ax=ax['A'],
        palette=sns.color_palette([orange]),
    )
    g.legend().remove()
    plt.setp(ax['A'].patches, linewidth=0)
    ax['A'].tick_params(axis='x', which='major', labelsize=7, labelrotation=90, pad=-3)
    ax['A'].set(xlabel=None, ylabel='PCE')

    dataset_indices1 = np.array([0])
    dataset_indices2 = np.floor(np.linspace(len(order) // 2, len(order) - 1, n_rel_diags - 1))
    dataset_indices = np.concatenate([dataset_indices1, dataset_indices2])
    dataset_indices = np.floor(np.linspace(0, len(order) - 1, n_rel_diags))

    rel_df = make_reliability_df(plot_df)
    palette = sns.color_palette()
    for dataset_idx, axis in zip(dataset_indices, [ax[letter] for letter in rel_diags_letters]):
        df_ds = rel_df.query('dataset_idx == @dataset_idx')
        colors = iter(palette)

        kwargs = dict(
            xyA=(dataset_idx, -0.17),
            xyB=(0.5, 1.2),
            coordsA=ax['A'].get_xaxis_transform(),
            coordsB=axis.transAxes,
            axesA=ax['A'],
            axesB=axis,
            connectionstyle='angle,angleA=0,angleB=90,rad=20',
        )

        con = ConnectionPatch(arrowstyle='->', linestyle='dotted', color='black', lw=1, **kwargs)
        head = ConnectionPatch(
            arrowstyle='-|>',
            lw=0,
            shrinkA=0,
            shrinkB=0,
            edgecolor='none',
            facecolor='black',
            linestyle='solid',
            **kwargs,
        )
        fig.add_artist(con)
        fig.add_artist(head)

        for name, df_model in df_ds.groupby('name'):
            plot_reliability_diagram(axis, df_model, agg_run=False, color=next(colors))
        (dataset,) = df_ds.dataset.unique()
        plot_consistency_bands_from_name(axis, dataset, config, coverage=0.9)
        axis.set_title(dataset)
        ax['B'].get_shared_y_axes().join(ax['B'], axis)
    for letter in rel_diags_letters[1:]:
        ax[letter].set_yticklabels([])

    fig.tight_layout()
    plot_or_savefig(path)


def hyp_test_hists(df, config, nb_test_samples=1000):
    ds_df = get_datasets_df(config, reload=True)
    test_statistics = {}
    progress = tqdm(df.groupby(['dataset', 'dataset_group']))
    n_runs = df.reset_index().run_id.nunique()
    for (dataset, dataset_group), df_by_ds in progress:
        n = ds_df.query('Dataset == @dataset')['Total instances']
        n *= config.dataset_groups[dataset_group].datamodule.args.train_inter_val_calib_test_split_ratio[-1]
        n = int(n)
        progress.set_postfix_str(f'{dataset} (size {n})')
        null_hyp = compute_null_hyp(nb_test_samples, n_runs, n)
        test_statistics[dataset] = null_hyp
    return test_statistics


def plot_hist_test_statistics(
    plot_df,
    config,
    order,
    test_statistics,
    metric='test_calib_l1',
    ncols=6,
    path=None,
):
    plot_df = plot_df[metric].reset_index()
    
    names = ['MIX-NLL', 'MIX-NLL + PCE-KDE', 'MIX-NLL + Recal', 'MIX-CRPS', 'SQR-CRPS']
    plot_df = plot_df.query('name in @names')

    plot_df['dataset'] = pd.Categorical(plot_df['dataset'], order.to_numpy())
    colors = sns.color_palette()

    groups = plot_df.groupby('dataset')
    size = len(groups)
    nrows = math.ceil(size / ncols)

    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3, nrows * 1.5),
        squeeze=False,
        sharex=False,
        sharey=True,
        dpi=200,
    )
    ax_flatten = ax.flatten()
    for i in range(size, len(ax_flatten)):
        ax_flatten[i].set_visible(False)
    ax_seq = iter(ax_flatten)

    for dataset, df_by_ds in groups:
        axis = next(ax_seq)
        null_hyp = test_statistics[dataset]

        axis.hist(null_hyp, bins=15, edgecolor='none')

        colors_seq = iter(colors)
        for model, df_model in df_by_ds.groupby('name'):
            color = next(colors_seq)
            test_statistic = df_model[metric].to_numpy().mean()
            axis.axvline(x=test_statistic, color=color, label=model)

        axis.set_title(dataset)

    for i in range(nrows):
        ax[i, 0].set(ylabel='Count')
    for i in range(ncols):
        ax_flatten[size - 1 - i].set(xlabel=rf'Test statistic')

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0),
        frameon=True,
        ncol=3,
        fontsize=14,
        title_fontsize=14,
    )

    fig.tight_layout()
    plot_or_savefig(path, fig)
