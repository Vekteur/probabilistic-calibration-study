from omegaconf import OmegaConf

from .callbacks import callbacks_config
from .dataset_groups import dataset_groups_config
from .experiments import configure_experiments
from .general import general_config
from .loggers import loggers_config
from .trainer import trainer_config
from .tuning import models_cls_config


def get_config(config=None):
    if config is None:
        config = OmegaConf.create()
    config_builders = [
        general_config,
        callbacks_config,
        trainer_config,
        dataset_groups_config,
        models_cls_config,
        loggers_config,
    ]
    for config_builder in config_builders:
        config = OmegaConf.merge(config_builder(config), config)
    configure_experiments(config)
    return config
