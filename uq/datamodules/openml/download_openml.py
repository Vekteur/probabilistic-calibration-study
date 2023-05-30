import pickle

import numpy as np
import openml
import pandas as pd
from tqdm import tqdm

from ..preprocessing import (
    InvalidDataset,
    load_dataset,
    remove_high_cardinality,
    remove_missing_values,
    remove_pseudo_categorical,
    save_dataset,
)


def download_openml_suite(suite_id, data_path):
    suite_path = data_path / 'openml' / str(suite_id)
    suite = openml.study.get_suite(suite_id)
    for task_id in tqdm(suite.tasks):
        try:
            download_openml_task(task_id, suite_path)
        except InvalidDataset:
            pass


def download_openml_task(task_id, suite_path):
    task = openml.tasks.get_task(task_id)
    label = task.target_name
    dataset = task.get_dataset()
    path = suite_path / dataset.name
    try:
        x, y = load_dataset(path)
        return x, y
    except FileNotFoundError:
        pass

    print(dataset.name, flush=True)

    assert label == dataset.default_target_attribute, (
        label,
        dataset.default_target_attribute,
    )
    x, y, categorical_mask, attribute_names = dataset.get_data(dataset_format='dataframe', target=label)
    x, y, categorical_mask = preprocess(x, y, categorical_mask)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    save_dataset(x, y, path)
    return x, y


def preprocess(x, y, categorical_mask):
    # For some datasets, y is a ndarray instead of a dataframe
    if type(y) == np.ndarray:
        y = pd.DataFrame(y)
    x, y, categorical_mask, n_high_cardinality = remove_high_cardinality(x, y, categorical_mask)
    x, y, n_pseudo_categorical = remove_pseudo_categorical(x, y)
    x, y, n_missing_cols, n_missing_rows = remove_missing_values(x, y)
    if x.columns.empty:   # get_dummies needs a dataframe that is not empty
        raise InvalidDataset('No remaining columns')
    x = pd.get_dummies(x)
    if x.columns.empty:
        raise InvalidDataset('No remaining columns')

    if x.columns.empty:
        raise InvalidDataset('No valid column')
    if len(x) < 100:
        print('Too few rows', flush=True)
        raise InvalidDataset('Too few rows in the dataset')
    # Categorical data could also be converted to one-hot
    x, y = x.to_numpy('float32'), y.to_numpy('float32')
    y = y.reshape(-1, 1)
    return x, y, categorical_mask
