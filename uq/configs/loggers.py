from omegaconf import OmegaConf


def loggers_config(config):
    return OmegaConf.create(dict(loggers={}))
