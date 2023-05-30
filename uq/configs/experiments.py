from omegaconf import OmegaConf

from uq import utils

log = utils.get_logger(__name__)


def qr(config):
    models_config = {key: value for key, value in config.models.items() if key.startswith('qr')}
    config.models = models_config


def uci(config):
    config.dataset_groups = dict(uci=config.dataset_groups.uci)


def meps(config):
    config.dataset_groups = dict(meps=config.dataset_groups.meps)


def configure_experiments(config):
    if config.name:
        if config.name == 'uci':
            uci(config)
        else:
            log.warn(f'The experiment {config.name} has no associated configuration.')
