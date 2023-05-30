"""
Code adapted from https://github.com/LeoGrin/tabular-benchmark/blob/main/data/data_utils.py
"""

import pickle

import pandas as pd


class InvalidDataset(Exception):
    pass


def remove_high_cardinality(X, y, categorical_mask, threshold=20):
    high_cardinality_mask = X.nunique() > threshold
    # print("high cardinality columns: {}".format(X.columns[high_cardinality_mask * categorical_mask]))
    n_high_cardinality = sum(categorical_mask * high_cardinality_mask)
    X = X.drop(X.columns[categorical_mask * high_cardinality_mask], axis=1)
    # print("Removed {} high-cardinality categorical features".format(n_high_cardinality))
    categorical_mask = [
        categorical_mask[i]
        for i in range(len(categorical_mask))
        if not (high_cardinality_mask[i] and categorical_mask[i])
    ]
    return X, y, categorical_mask, n_high_cardinality


def remove_pseudo_categorical(X, y):
    """Remove columns where most values are the same"""
    num_cols = set(X.select_dtypes(include='number').columns)
    num_mask = X.columns.isin(num_cols)
    pseudo_categorical_cols_mask = X.nunique() < 10
    X = X.drop(X.columns[num_mask & pseudo_categorical_cols_mask], axis=1)
    return X, y, sum(pseudo_categorical_cols_mask)


def remove_missing_values(X, y, threshold=0.2):
    """Remove columns where most values are missing, then remove any row with missing values"""
    missing_cols_mask = pd.isnull(X).mean(axis=0) > threshold
    X = X.drop(X.columns[missing_cols_mask], axis=1)
    missing_rows_mask = pd.isnull(X).any(axis=1)
    X = X[~missing_rows_mask]
    y = y[~missing_rows_mask]
    return X, y, sum(missing_cols_mask), sum(missing_rows_mask)


def save_dataset(x, y, path):
    with open(path / 'x.npy', 'wb') as f:
        pickle.dump(x, f)
    with open(path / 'y.npy', 'wb') as f:
        pickle.dump(y, f)


def load_dataset(path):
    with open(path / 'x.npy', 'rb') as f:
        x = pickle.load(f)
    with open(path / 'y.npy', 'rb') as f:
        y = pickle.load(f)
    return x, y
