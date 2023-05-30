from functools import partial

from omegaconf import OmegaConf

from uq.datamodules.openml.openml_module import OpenMLDataModule
from uq.datamodules.toy.toy_module import ToyDataModule
from uq.datamodules.uci.uci_module import UCIDataModule
from uq.utils.general import filter_dict


def default_args(config):
    return dict(
        batch_size=config.default_batch_size,
        train_inter_val_calib_test_split_ratio=[0.65, 0.0, 0.1, 0.15, 0.1],
        num_workers=0,
        pin_memory=False,
    )


def uci_config(config):
    names = [
        'CPU',
        'Yacht',
        'MPG',
        'Energy',
        'Crime',
        'Fish',
        'Concrete',
        'Airfoil',
        'Kin8nm',
        'Power',
        'Naval',
        'Protein',
    ]
    datamodule_config = dict(
        cls=UCIDataModule,
        args=default_args(config),
    )
    return dict(
        names=names,
        datamodule=datamodule_config,
    )


def openml297_config(config):
    names = [
        'wine_quality',
        'isolet',
        'cpu_act',
        'sulfur',
        'Brazilian_houses',
        'Ailerons',
        'MiamiHousing2016',
        'pol',
        'elevators',
        'Bike_Sharing_Demand',
        'fifa',
        #'houses', # Duplicate of california
        'california',
        'superconduct',
        'house_sales',
        'house_16H',
        'diamonds',
        'medical_charges',
        'year',
        'nyc-taxi-green-dec-2016',
    ]
    datamodule_config = dict(
        cls=partial(OpenMLDataModule, suite_id=297),
        args=default_args(config),
    )
    return dict(
        names=names,
        datamodule=datamodule_config,
    )


def openml299_config(config):
    names = [
        'analcatdata_supreme',
        'Mercedes_Benz_Greener_Manufacturing',
        'visualizing_soil',
        'yprop_4_1',
        'OnlineNewsPopularity',
        'black_friday',
        'SGEMM_GPU_kernel_performance',
        'particulate-matter-ukair-2017',
    ]
    datamodule_config = dict(
        cls=partial(OpenMLDataModule, suite_id=299),
        args=default_args(config),
    )
    return dict(
        names=names,
        datamodule=datamodule_config,
    )


def openml269_config(config):
    names = [
        'tecator',
        'boston',
        #'sensory', # Only has categorical columns with high cardinality
        'MIP-2016-regression',
        'socmob',
        'Moneyball',
        'house_prices_nominal',
        'us_crime',
        'quake',
        'space_ga',
        'abalone',
        'SAT11-HAND-runtime-regression',
        'Santander_transaction_value',
        #'QSAR-TID-11', # Too high dimensional (X.shape[1] too high)
        #'QSAR-TID-10980',
        'colleges',
        'topo_2_1',
        'Allstate_Claims_Severity',
        'Yolanda',
        'Buzzinsocialmedia_Twitter',
        'Airlines_DepDelay_10M',
    ]
    datamodule_config = dict(
        cls=partial(OpenMLDataModule, suite_id=269),
        args=default_args(config),
    )
    return dict(
        names=names,
        datamodule=datamodule_config,
    )


def toy_config(config):
    sizes = [2**i for i in range(9, 18)]
    names = [f'toy_{size}' for size in sizes]
    args = dict(
        batch_size=512,
        train_inter_val_calib_test_split_ratio=[0.01, 0.01, 0.01, 0.01, 0.96],
        num_workers=0,
        pin_memory=False,
    )
    datamodule_config = dict(
        cls=ToyDataModule,
        args=args,
    )
    return dict(
        names=names,
        datamodule=datamodule_config,
    )


def dataset_groups_config(config):
    dataset_groups_config = dict(
        uci=uci_config(config),
        oml_297=openml297_config(config),
        oml_299=openml299_config(config),
        oml_269=openml269_config(config),
        toy=toy_config(config),
    )

    if config.get('selected_dataset_groups') is not None:
        dataset_groups_config = filter_dict(dataset_groups_config, config.selected_dataset_groups)

    return OmegaConf.create(
        dict(dataset_groups=dataset_groups_config),
        flags={'allow_objects': True},
    )
