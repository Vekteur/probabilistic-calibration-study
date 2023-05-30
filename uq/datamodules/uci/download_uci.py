import io
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import openml as oml
import pandas as pd
from tqdm import tqdm

from ..preprocessing import InvalidDataset, load_dataset, remove_missing_values, save_dataset

urls = {
    'Airfoil': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat',
    'Boston': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
    'Concrete': 'http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
    'CPU': 'https://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data',
    'Crime': 'https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data',
    'Energy': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
    'Fish': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv',
    'Kin8nm': None,
    'MPG': 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
    'Naval': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip',
    'Power': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
    'Protein': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
    'Wine_Red': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
    'Wine_White': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
    'Yacht': 'http://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
    'Year': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
}


def put_last(df, column_idx):
    l = list(df.columns.values)
    col = l.pop(column_idx)
    l.append(col)
    return df[l]


def download_uci_df(name, url):
    if name == 'Airfoil':
        return pd.read_table(url, delim_whitespace=True, header=None)
    elif name == 'Boston':
        return pd.read_table(url, delim_whitespace=True, header=None)
    elif name == 'Concrete':
        return pd.read_excel(url)
    elif name == 'CPU':
        return pd.read_csv(url, header=None, delimiter=',')
    elif name == 'Crime':
        return pd.read_csv(url, header=None, delimiter=',', na_values='?')
    elif name == 'Energy':
        return pd.read_excel(url)
    elif name == 'Fish':
        return pd.read_csv(url, header=None, delimiter=';')
    elif name == 'Kin8nm':
        ds = oml.datasets.get_dataset(189)
        X, y, _, _ = ds.get_data(dataset_format='dataframe', target=ds.default_target_attribute)
        return pd.concat((X, y.to_frame()), axis='columns')
    elif name == 'MPG':
        df = pd.read_table(url, delim_whitespace=True, header=None, na_values='?')
        return put_last(df, 0)
    elif name == 'Naval':
        r = urlopen(url).read()
        with ZipFile(io.BytesIO(r)) as zip:
            with zip.open('UCI CBM Dataset/data.txt') as f:
                return pd.read_table(f, delim_whitespace=True, header=None)
    elif name == 'Power':
        r = urlopen(url).read()
        with ZipFile(io.BytesIO(r)) as zip:
            with zip.open('CCPP/Folds5x2_pp.xlsx') as f:
                return pd.read_excel(f)
    elif name == 'Protein':
        df = pd.read_csv(url, header='infer', delimiter=',')
        return put_last(df, 0)
    elif name == 'Wine_Red':
        return pd.read_csv(url, header='infer', delimiter=';')
    elif name == 'Wine_White':
        return pd.read_csv(url, header='infer', delimiter=';')
    elif name == 'Yacht':
        return pd.read_table(url, delim_whitespace=True, header=None)
    elif name == 'Year':
        df = pd.read_csv(url, header=None)
        return put_last(df, 0)


def preprocess(x, y):
    x, y, n_missing_cols, n_missing_rows = remove_missing_values(x, y)

    # Only keep numeric features
    x = x.select_dtypes(include='number')

    if x.columns.empty:
        raise InvalidDataset('No valid column')
    # Categorical data could also be converted to one-hot
    x, y = x.to_numpy('float32'), y.to_numpy('float32')
    y = y.reshape(-1, 1)
    return x, y


def download_uci(name, url, data_path):
    path = data_path / 'uci' / name
    try:
        x, y = load_dataset(path)
        return x, y
    except FileNotFoundError:
        pass

    df = download_uci_df(name, url)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    x, y = preprocess(x, y)

    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    save_dataset(x, y, path)
    return x, y


def download_all_uci(data_path):
    for name, url in tqdm(urls.items()):
        print(name, flush=True)
        download_uci(name, url, data_path)


if __name__ == '__main__':
    for name, url in urls.items():
        print(name)
        df = download_uci_df(name, url)
        print(df.dtypes)
        print(df.columns[:8])
