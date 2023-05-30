from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

from uq.datamodules.toy.toy_module import toy_cond_dist
from uq.utils.checkpoints import load_datamodule, load_rc_checkpoint
from uq.utils.dist import icdf
from uq.utils.general import filter_dict

from .plot_metrics import make_plot_df, plot_metric_comparison_per_epoch


def model_name(model, hparams):
    hparams_str = ','.join(f'{key}={value}' for key, value in hparams.items())
    return f'{model}({hparams_str})'


def get_unique_hparams(hparams_list):
    from collections import defaultdict

    count = defaultdict(set)
    for hparams in hparams_list:
        for hparam, value in hparams.items():
            count[hparam].add(value)
    unique_hparams = {}
    variable_hparams = []
    for hparam, values in count.items():
        if len(values) == 1:
            unique_hparams[hparam] = next(iter(values))
        else:
            variable_hparams.append(hparam)
    return unique_hparams, variable_hparams


def make_model_name_by_hparams(df):
    df = df.copy()
    assert len(df) != 0
    _, variable_hparams = get_unique_hparams(df.hparams)

    def get_model_name(d):
        return model_name(d.model, filter_dict(d.hparams, variable_hparams))

    df.model = df.apply(lambda d: get_model_name(d), axis=1)
    df = df.sort_values('model', kind='stable')
    if 'lambda_' in df.columns:
        df = df.sort_values('lambda_', kind='stable')
    return df


def get_cond_dist(config, dataset_group, dataset):
    if dataset_group == 'toy':
        size = 10000
        x, cond_dist = toy_cond_dist(size)
    else:
        rc = load_rc_checkpoint(
            config=config,
            dataset_group=dataset_group,
            dataset=dataset,
        )
        datamodule = load_datamodule(rc)
        datamodule.load_datasets()
        cond_dist = datamodule.cond_dist
    return cond_dist


def get_true_sharpness(config, dataset_group, dataset):
    cond_dist = get_cond_dist(config, dataset_group, dataset)
    stddev = cond_dist.stddev
    length = icdf(cond_dist, torch.tensor(0.95)[None, None]) - icdf(cond_dist, torch.tensor(0.05)[None, None])
    return stddev.mean().numpy(), length.mean().numpy()


def plot_calib_and_sharpness_metrics_per_epoch(
    model_df,
    cond_stddev=None,
    cond_interval_length=None,
    posthoc=False,
    only_quantile=False,
    metrics=None,
    **kwargs,
):

    unique_hparams, _ = get_unique_hparams(model_df.hparams)
    title = ','.join(f'{key}={value}' for key, value in unique_hparams.items())
    model_df = make_model_name_by_hparams(model_df)
    df_plot = make_plot_df(model_df)
    if metrics is None:
        metrics = [
            'train_nll',
            'train_calib_l1',
            'train_stddev',
            'val_nll',
            'val_calib_l1',
            'val_stddev',
            'train_wis',
            'val_coverage_90',
            'val_length_90',
            'val_wis',
            'val_mse',
            'val_mae',
            'val_calib_l2',
            'val_calib_kl',
            'val_pearson',
            'val_base_loss',
        ]
    if posthoc:
        metrics.extend(
            [
                'val_posthoc_nll',
                'val_posthoc_calib_l1',
                'val_posthoc_stddev',
                'val_posthoc_wis',
                'val_posthoc_coverage_90',
                'val_posthoc_length_90',
                'val_posthoc_mse',
                'val_posthoc_mae',
                'val_posthoc_pearson',
            ]
        )
    if only_quantile:
        ignore = ['train_nll', 'val_nll', 'val_posthoc_nll', 'val_calib_kl']
        metrics = [metric for metric in metrics if metric not in ignore]
    df_plot = df_plot.query('metric in @metrics')

    # Make sure that the metrics are displayed in the right order
    d = {metric: order for order, metric in enumerate(metrics)}
    df_plot['order'] = df_plot['metric'].map(d)
    df_plot = df_plot.sort_values('order', kind='stable')
    df_plot = df_plot.drop(columns='order')

    fig = plot_metric_comparison_per_epoch(df_plot, cmap='magma', ncols_legend=1, **kwargs)

    for metric, order in d.items():
        if '_coverage_90' in metric:
            fig.axes[order].axhline(0.9, color='black', linestyle=':')
        if '_stddev' in metric and cond_stddev is not None:
            fig.axes[order].axhline(cond_stddev, color='black', linestyle=':')
        if metric == 'val_length_90' and cond_interval_length is not None:
            fig.axes[order].axhline(cond_interval_length, color='black', linestyle=':')
    fig.legends[0].set_title(f'Model ({title})')
    fig.tight_layout()
    return fig
