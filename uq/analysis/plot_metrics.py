import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns

from .dataframes import get_all_metrics


def plot_unit(x, y, path, name, unit='epoch'):
    # Using `fig, axis = plt.subplots()` results in a memory leak
    fig = mpl.figure.Figure()
    axis = fig.subplots()
    axis.plot(x, y)
    axis.set(xlabel=unit, ylabel=name, title=f'{name} per {unit} during training')
    fig.savefig(path / f'{name}.png')
    plt.close(fig)


def set_base_loss_column(df, stage='val'):
    df['ID'] = df.index
    df[f'{stage}_base_loss'] = stage + '_' + df['base_loss']
    base_losses = df[f'{stage}_base_loss'].unique()
    only_base_losses = df[base_losses]
    stacked = only_base_losses.stack().rename('scoring')
    df[f'{stage}_base_loss'] = df.join(stacked, on=['ID', f'{stage}_base_loss'], how='left').scoring
    df.drop(columns=['ID'], inplace=True)


def make_plot_df(df):
    """
    Transform the dataframe to a dataframe with columns
    ('dataset_group', 'dataset', 'model', 'run_id', 'metric', 'epoch')
    """
    df = df.copy()
    train_val_metrics = {
        metric
        for metric in get_all_metrics(df)
        if (metric.startswith('train_') or metric.startswith('val_'))
        and not (metric.startswith('train_quantile_score_') or metric.startswith('val_quantile_score_'))
    }
    columns_to_concat = []
    for metric in train_val_metrics:
        columns_to_concat.append(df.metrics.map(lambda d: d['per_epoch'][metric]).rename(metric))
    df = pd.concat([df] + columns_to_concat, axis=1)
    df = df.copy()   # copy is used to defragment the dataframe
    set_base_loss_column(df, stage='train')
    set_base_loss_column(df, stage='val')
    df = df.drop(columns='base_loss')

    df = df.set_index(['dataset_group', 'dataset', 'model', 'run_id'])
    df = df[list(train_val_metrics)].stack()
    df = df.rename_axis(index={None: 'metric'})
    df = df.apply(lambda x: pd.Series(x, dtype=float)).stack()
    df = df.rename_axis(index={None: 'epoch'})
    df = df.to_frame(name='value').reset_index()
    df = df.sort_values('metric', kind='stable')
    return df


# Easy to do with seaborn byt doing it with matplotlib is more easily customizable and a lot faster
def plot_dataset_seaborn(df):
    df_plot = make_plot_df(df)
    g = sns.FacetGrid(
        df_plot,
        row='metric',
        col='model',
        sharex=False,
        sharey=False,
        margin_titles=True,
    )
    g.map(sns.lineplot, 'epoch', 'value', ci=90)


def sem(x):
    return scipy.stats.sem(x, ddof=0)


def plot_sem(axis, df, label=None, color=None):
    axis.plot(df.index, df['mean'], label=label, color=color)
    axis.fill_between(
        df.index,
        df['mean'] - df['sem'],
        df['mean'] + df['sem'],
        alpha=0.2,
        color=color,
        zorder=10,
    )


def plot_all_runs(axis, df, label=None, color=None):
    for run_id, df_run in df.groupby('run_id'):
        label = label if run_id == 0 else None
        axis.plot(df_run['epoch'], df_run['value'], label=label, color=color, lw=0.5)


def plot_agg_runs(axis, df, label=None, color=None):
    df = df.drop(columns='run_id').groupby('epoch').agg(['mean', sem])
    df = df.droplevel(0, axis=1)   # Remove the 'value' level
    plot_sem(axis, df, label=label, color=color)


def plot_runs(*args, agg_run=True, **kwargs):
    if agg_run:
        plot_agg_runs(*args, **kwargs)
    else:
        plot_all_runs(*args, **kwargs)


def plot_hline(axis, df, df_best_iter, color=None, agg_run=True):
    df_best_iter = df_best_iter.reset_index()[['run_id', 'epoch']]
    df_merged = pd.merge(df_best_iter, df)
    ys = df_merged.value
    if agg_run:
        ys = [ys.mean()]
    for y in ys:
        axis.axhline(y, color=color, linestyle='--', lw=1, zorder=100)


def plot_vline(axis, iters, color=None, agg_run=True):
    if agg_run:
        iters = [iters.mean()]
    for iter in iters:
        axis.axvline(iter, color=color, linestyle='--', lw=1, zorder=100)


def plot_runs_and_lines(axis, df_plot, df_best_iter, label=None, color=None, agg_run=True):
    plot_runs(axis, df_plot, label=label, color=color, agg_run=agg_run)
    plot_vline(axis, df_best_iter, color=color, agg_run=agg_run)
    plot_hline(axis, df_plot, df_best_iter, color=color, agg_run=agg_run)


def get_best_iter(df, model_name):
    df = df.query(f'metric == "val_base_loss" and model == "{model_name}"')
    df_best_iter = df.loc[df.groupby('run_id')['value'].idxmin()]
    return df_best_iter.set_index(['model', 'run_id', 'metric']).epoch   # , df_best_iter.value.mean()


# Ad hoc function
def get_colors(cmap, models):
    import re

    def get_lambda(model):
        match = re.search(r'lambda_=(\d+)', model)
        if match is None:
            return None
        return int(match.group(1))

    lambdas = set(get_lambda(model) for model in models)
    if len(lambdas) == 1:
        cmap = None

    if cmap is None:
        colors = mpl.colors.TABLEAU_COLORS
        assert len(models) <= len(colors)
    elif 0 in lambdas:
        cm = mpl.cm.get_cmap(cmap)
        color_linear = np.linspace(1, 0, len(models) + 1)[:-1]
        colors = [cm(l) for l in color_linear]
        colors[0] = 'green'
    else:
        cm = mpl.cm.get_cmap(cmap)
        color_linear = np.linspace(1, 0, len(models) + 2)[1:-1]
        colors = [cm(l) for l in color_linear]
    return colors


def plot_metric_comparison_per_epoch(df, agg_run=True, cmap=None, ncols=3, ncols_legend=3):
    fixed_columns = ['dataset_group', 'dataset']
    for col in fixed_columns:
        assert df[col].nunique() == 1
    df = df.drop(columns=fixed_columns)
    df = df.sort_values('epoch', kind='stable')
    plot_both = agg_run == 'both'
    if plot_both:
        agg_run = False

    metrics = df['metric'].unique()
    models = df['model'].unique()
    colors = get_colors(cmap, models)

    size = len(metrics)
    nrows = math.ceil(size / ncols)
    fig, ax = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 4, nrows * 3),
        squeeze=False,
        sharex=True,
        dpi=200,
    )
    ax = ax.flatten()
    for i in range(size, len(ax)):
        ax[i].set_visible(False)

    left_base_losses = ['nll']
    right_base_losses = ['crps', 'expected_qs']

    def get_base_loss(model):
        for base_loss in left_base_losses + right_base_losses:
            if base_loss in model:
                return base_loss

    base_losses = set(get_base_loss(model) for model in models)

    model_to_best_iter = {model_name: get_best_iter(df, model_name) for model_name in models}

    for metric_id, metric_name in enumerate(metrics):
        axis = ax[metric_id]
        use_twins = False
        if 'base_loss' in metric_name:
            if len(base_losses) > 1:
                use_twins = True
                twin_axis = axis.twinx()
                axis.set_ylabel(' / '.join(left_base_losses))
                twin_axis.set_ylabel(' / '.join(right_base_losses))
            else:
                axis.set_ylabel(next(iter(base_losses)))
        for model_id, (model_name, color) in enumerate(zip(models, colors)):
            df_plot = df.query(f'metric == "{metric_name}" and model == "{model_name}"')
            df_best_iter = model_to_best_iter[model_name]
            orig_axis = axis
            if use_twins:
                for base_loss in right_base_losses:
                    if base_loss in model_name:
                        axis = twin_axis
                        break
            df_plot = df_plot.drop(columns=['metric', 'model'])
            if not df_plot.empty:
                plot_runs_and_lines(
                    axis,
                    df_plot,
                    df_best_iter,
                    label=model_name,
                    color=color,
                    agg_run=agg_run,
                )
                if plot_both:
                    plot_runs_and_lines(
                        axis,
                        df_plot,
                        df_best_iter,
                        label=None,
                        color='black',
                        agg_run=not agg_run,
                    )
            axis = orig_axis
        axis.set_title(metric_name, fontsize=15)
        axis.set_xlabel('epoch')
        axis.margins(x=0.02)
        if '_calib_l' in metric_name or '_length_' in metric_name or '_stddev' in metric_name:
            axis.set_ylim(bottom=0)
        if 'loss' not in metric_name:
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
    return fig


def plot_metric_comparison_per_epoch_with_models_as_columns(df, agg_run=True, filter_runs=None):
    fixed_columns = ['dataset_group', 'dataset']
    for col in fixed_columns:
        assert df[col].nunique() == 1
    df = df.drop(columns=fixed_columns)

    if filter_runs is not None:
        df = df[df['run_id'].isin(filter_runs)]
    metrics = df['metric'].unique()
    models = df['model'].unique()
    fig, ax = plt.subplots(
        len(metrics),
        len(models),
        figsize=(len(models) * 5, len(metrics) * 3),
        squeeze=False,
        sharex='col',
        sharey='row',
    )
    for metric_id, metric_name in enumerate(metrics):
        for model_id, model_name in enumerate(models):
            df_plot = df.query(f'metric == "{metric_name}" and model == "{model_name}"')
            df_plot = df_plot.drop(columns=['metric', 'model'])
            axis = ax[metric_id][model_id]
            axis.margins(x=0.02)
            if agg_run:
                plot_agg_runs(axis, df_plot)
            else:
                plot_all_runs(axis, df_plot)

    for metric_id, metric_name in enumerate(metrics):
        ax[metric_id, 0].set_ylabel(
            metric_name,
            fontweight='bold',
            fontsize=16,
            rotation=90,
            va='bottom',
        )
    for model_id in range(len(models)):
        ax[-1, model_id].set_xlabel('epoch', fontsize=12)
    for row in range(0, len(metrics), 5):
        for model_id, model_name in enumerate(models):
            ax[row, model_id].set_title(model_name, fontweight='bold', fontsize=18, va='bottom')
    fig.suptitle(
        'Comparison of metrics per model and epoch',
        fontweight='bold',
        fontsize=24,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.96)
    return fig
