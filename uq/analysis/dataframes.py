import dataclasses
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import yaml
from omegaconf import OmegaConf

from uq.configs.config import get_config
from uq.train import instantiate
from uq.utils.general import filter_dict
from uq.utils.run_config import RunConfig


# Small hack that allows to pickle the logs even if some pickled class does not exist anymore.
# In that case, the pickled class can not be loaded but we only care about the metrics.
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except AttributeError:
            return object


def load_config(path):
    with open(Path(path) / 'config.yaml', 'r') as f:
        return OmegaConf.create(yaml.load(f, Loader=yaml.Loader), flags={'allow_objects': True})


def make_df(config, dataset_group, dataset, reload=True):
    dataset_path = Path(config.log_dir) / dataset_group / dataset
    if not dataset_path.exists():
        return None
    df_path = dataset_path / 'df.pickle'
    if not reload and df_path.exists():
        with open(df_path, 'rb') as f:
            return CustomUnpickler(f).load()
    series_list = []
    for run_config_path in dataset_path.rglob('run_config.pickle'):
        with open(run_config_path, 'rb') as f:
            rc = CustomUnpickler(f).load()
        series_list.append(rc.to_series())
    if len(series_list) == 0:
        return None
    df = pd.concat(series_list, axis=1).T
    with open(df_path, 'wb') as f:
        pickle.dump(df, f)
    return df


def load_df(config, dataset_group=None, dataset=None, tuning=None, reload=True):
    assert config is not None
    dfs = []
    if dataset_group is None:
        dataset_groups = config.dataset_groups.keys()
    else:
        dataset_groups = [dataset_group]
    for curr_dataset_group in dataset_groups:
        if dataset is None:
            datasets = config.dataset_groups[curr_dataset_group].names
        else:
            datasets = [dataset]
        for curr_dataset in datasets:
            df = make_df(config, curr_dataset_group, curr_dataset, reload=reload)
            if df is not None:
                dfs.append(df)
    if not dfs:
        raise RuntimeError('Dataframe not found')
    df = pd.concat(dfs)
    if tuning is not None:
        df = df[df['tuning'] == tuning]
    return df


def union(sets):
    res = set()
    for s in sets:
        res |= s
    return res


def get_all_metrics(df):
    return df.metrics.map(lambda d: set(d['per_epoch'].keys())).agg(union)


def is_test_metric(metric, remove_quantile_scores=False):
    return (
        metric.startswith('test_')
        and not (remove_quantile_scores and metric.startswith('test_quantile_score_'))
    ) or metric in ['val_calib_l1', 'val_wis', 'val_nll']


def get_test_metrics(df, remove_quantile_scores=False):
    return [metric for metric in get_all_metrics(df) if is_test_metric(metric, remove_quantile_scores)]


def build_test_metric_accessor(metric):
    def impl(df):
        epoch_values = df.metrics['per_epoch'][metric]
        if len(epoch_values) == 0:
            return np.nan
        elif len(epoch_values) == 1:
            return next(iter(epoch_values.values()))
        raise RuntimeError('More than one value is available')

    return impl


def get_hparams(df):
    hparams = df.hparams.map(lambda d: set(d.keys())).agg(union)
    # Strangely when there are no hparams, agg returns a Series instead of a set
    if type(hparams) == pd.Series:
        return set()
    return hparams


def build_hparam_accessor(hparam):
    return lambda df: df.hparams.get(hparam, np.nan)


def build_batchsize_accessor(config):
    def batchsize_accessor(df):
        fields = map(lambda f: f.name, dataclasses.fields(RunConfig))
        d = filter_dict(df.to_dict(), fields)
        rc = RunConfig(**d)
        rc.config = config
        return rc.dataset_group_config.datamodule.args.get('batch_size', np.nan)

    return batchsize_accessor


def get_hparams_accessors(hparams, config):
    hparams_accessors = {}
    hparams_accessors['batch_size'] = build_batchsize_accessor(config)
    hparams_accessors.update({hparam: build_hparam_accessor(hparam) for hparam in hparams})
    return hparams_accessors


def set_hparams_columns(df, config, hparams=None):
    if hparams is None:
        hparams = get_hparams(df)
    accessors = get_hparams_accessors(hparams, config)
    for key, accessor in accessors.items():
        df[key] = df.apply(accessor, axis=1)


def set_test_metrics_columns(df, test_metrics=None, add_infos=True):
    if test_metrics is None:
        test_metrics = get_test_metrics(df)
    for metric in test_metrics:
        df[metric] = df.apply(build_test_metric_accessor(metric), axis=1)
    general_metrics = [
        'best_iter',
        'best_score',
        'train_time',
        'val_time',
        'test_time',
    ]
    kept_metrics = []
    if add_infos:
        for metric in general_metrics:
            try:
                df[metric] = df.metrics.map(lambda d: d[metric])
                kept_metrics.append(metric)
            except KeyError:
                pass
    return list(test_metrics) + kept_metrics
    # if add_last_epoch and len(test_metrics) > 0:
    #     first_metric = next(iter(test_metrics))
    #     df['last_epoch'] = df.metrics.map(lambda d: list(d[first_metric].keys())[-1])


def agg_constant(d):
    distinct_values = np.unique(d.values)
    if len(distinct_values) == 1:
        return distinct_values[0]
    return np.nan


def agg_mean_std(x):
    mean = np.mean(x)
    std = None
    if len(x) > 1:
        std = scipy.stats.sem(x, ddof=1)
    return (mean, std)


def format_cell(x):
    mean, sem = x
    if np.isnan(mean):
        return 'NA'
    s = f'{mean:#.3}'
    if sem is not None:
        sem = float(sem)
        s += rf' +- {sem:#.2}'
    return s


def agg_mean_std_format(x):
    return format_cell(agg_mean_std(x))


def make_test_df(
    df,
    config,
    groupby=['dataset_group', 'dataset', 'model'],
    average_mode=None,
):
    """
    Transform the dataframe to a dataframe with columns
    ('dataset_group', 'dataset', 'model', 'run_id') + all_test_metrics + all_hparams.
    It is optionally aggregated over run_id.
    """
    df = df.copy()
    assert len(df) > 0, 'The dataframe must not be empty'
    assert average_mode in [None, 'mean_std_format', 'mean']
    hparams = get_hparams(df)
    set_hparams_columns(df, config, hparams)
    test_metrics = get_test_metrics(df)
    cols_to_average = set_test_metrics_columns(df, test_metrics, add_infos=True)
    # df['test_calib_l2'] *= 1000
    cols_to_keep_constant = [hparam for hparam in hparams if hparam not in groupby]
    df = df.drop(columns=['config', 'metrics', 'hparams'])
    if average_mode is not None:
        agg_dict = {}
        # Take the mean of the metrics
        agg_mean_fn = {'mean_std_format': agg_mean_std, 'mean': np.mean}[average_mode]
        for col in cols_to_average:
            agg_dict[col] = agg_mean_fn
        # Keep the hyperparameters constant
        for col in cols_to_keep_constant:
            agg_dict[col] = agg_constant
        df = df.drop(columns='run_id').groupby(groupby, sort=False, dropna=False)
        # Add the size of the aggregation as a column
        df_size = df.agg({col: 'size'})
        df = df.agg(agg_dict)
        df['size'] = df_size[col]
        if average_mode == 'mean_std_format':
            for col in cols_to_average:
                df[col] = df[col].apply(format_cell)
    else:
        groupby += ['run_id']
    df = df.reset_index().convert_dtypes().set_index(groupby)
    ordered_columns = list(cols_to_keep_constant) + sorted(cols_to_average)
    if average_mode is not None:
        ordered_columns.append('size')
    df = df[ordered_columns]
    df = df.sort_values(groupby, kind='stable')
    return df


def make_test_df_for_tuning(df=None, config=None, groupby=None, average_mode=None):
    path = Path(config.log_dir) / f'test_df_tuning_{average_mode}.pickle'
    if df is None:
        assert path.exists()
        return pd.read_pickle(path)
    if groupby is None:
        groupby = ['dataset_group', 'dataset', 'model'] + list(get_hparams(df))
    test_df = make_test_df(df, config, groupby=groupby, average_mode=average_mode)
    test_df.to_pickle(path)
    return test_df


def separate_posthoc(df):
    df_no_posthoc = df.copy()
    df_posthoc = df.copy()
    cols = df.columns
    posthoc_cols = [col for col in cols if '_posthoc_' in col]
    posthoc_cols_map = {col: col.replace('_posthoc_', '_') for col in posthoc_cols}
    no_posthoc_cols = set(posthoc_cols_map.values())
    df_no_posthoc = df_no_posthoc[[col for col in cols if col not in posthoc_cols]]
    df_posthoc = df_posthoc[[col for col in cols if col not in no_posthoc_cols]]
    df_posthoc = df_posthoc.rename(columns=posthoc_cols_map)
    df_no_posthoc['posthoc'] = False
    df_posthoc['posthoc'] = True
    df = pd.concat((df_no_posthoc, df_posthoc)).set_index('posthoc', append=True)
    index = df.index.names
    df = df.reset_index()
    df['full_model'] = df.apply(
        lambda d: d['model'] + (' posthoc' if d['posthoc'] else ''),
        axis='columns',
    )
    df = df.set_index(index).set_index('full_model', append=True)
    return df


def compute_datasets_df(config):
    data = defaultdict(list)
    for dataset_group, dataset_group_config in config.dataset_groups.items():
        for dataset in dataset_group_config.names:
            rc = RunConfig(
                config=config,
                dataset_group=dataset_group,
                dataset=dataset,
                model=None,
            )
            datamodule = instantiate(
                rc.dataset_group_config.datamodule,
                rc=rc,
                name=rc.dataset,
                seed=2000 + rc.run_id,
            )
            datamodule.load_datasets()
            data_train = datamodule.data_train
            nb_instances = datamodule.total_size
            first_item = next(iter(data_train))
            x, y = first_item
            nb_features = x.shape[0]
            description = {
                'Group': dataset_group,
                'Dataset': dataset,
                'Total instances': nb_instances,
                'Nb of features': nb_features,
                'Ratio': nb_instances / nb_features,
            }
            for key, value in description.items():
                data[key].append(value)
    return pd.DataFrame(data).set_index(['Group', 'Dataset'])


def get_datasets_df(config, reload=False):
    path = Path(config.log_dir) / 'datasets_df.pickle'
    if not reload and path.exists():
        return pd.read_pickle(path)
    df = compute_datasets_df(config)
    df.to_pickle(path)
    return df


def make_df_abb(datasets):
    df_abb = pd.DataFrame({'dataset': datasets}).sort_values('dataset')
    df_abb['abb'] = df_abb['dataset'].str[:3].str.upper()

    def agg(x):
        x = x['abb']
        if len(x) > 1:
            x = x.str[:-1]
            x += np.arange(1, len(x) + 1).astype(str)
        return x

    df_abb['abb'] = df_abb.groupby('abb').apply(agg).droplevel(0)
    return df_abb.sort_values('dataset')


def debug_merge(baseline, compared, join_by):
    from IPython.display import display

    baseline = baseline.head(10)
    compared = compared.head(10)
    merged = compared.merge(baseline, how='left', on=join_by)
    display(baseline)
    display(compared)
    display(merged)


def build_grouped_comparison_df(
    df,
    metrics,
    baseline_query,
    join_by,
    columns_to_keep,
    isolate_baseline=True,
):
    """
    The returned dataframe is grouped by join_by + columns_to_keep
    """
    df = df[metrics].stack().rename_axis(index={None: 'metric'}).to_frame(name='value')
    # Build the baseline and compared dataframes
    baseline = df.query(baseline_query)
    compared = df
    if isolate_baseline:
        compared = compared.query(f'not ({baseline_query})')
    compared = compared.reset_index(level=columns_to_keep)
    baseline = baseline.rename(columns={'value': 'baseline'})
    compared = compared.rename(columns={'value': 'compared'})
    # Merge the baseline and compared dataframes
    # debug_merge(baseline, compared, join_by)
    merged = compared.merge(baseline, how='left', on=join_by, validate='many_to_one')
    if 'run_id' in merged.index.names:
        merged = merged.reset_index(level='run_id', drop=True)
    groupby = merged.index.names + columns_to_keep
    return merged.groupby(groupby, dropna=False)


# Workaround because fillna -1 does not work on string columns
def fillna(df, float_value=-1, str_value=''):
    df = df.copy()
    index = df.index.names
    df = df.reset_index()
    float_cols = df.select_dtypes('number').columns
    str_cols = df.select_dtypes('string').columns
    df[float_cols] = df[float_cols].fillna(float_value)
    df[str_cols] = df[str_cols].fillna(str_value)
    if df.index.names != [None]:
        df = df.set_index(index)
    return df
