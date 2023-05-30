import json
import logging

from omegaconf import OmegaConf

from uq.train import get_model_args, load_datamodule
from uq.utils.general import instantiate
from uq.utils.run_config import RunConfig

log = logging.getLogger('uq')


def get_hparams(rc, config):
    if config.tuning:   # If there was hyperparameter tuning before the run
        hparams_path = rc.model_path / 'best_hparams.json'
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)
    else:
        hparams = {}
    return hparams


def get_checkpoint_path(rc, epoch=None):
    if epoch == 'last':
        ckpt_name = 'last.ckpt'
    elif epoch == 'best':
        ckpts_list = []
        for ckpt_path in rc.checkpoints_path.iterdir():
            if ckpt_path.name != 'last.ckpt':
                ckpts_list.append(ckpt_path.name)
        ckpts_list.sort(key=lambda x: int(x[6:10]))
        if len(ckpts_list) != 1:
            log.warn(f'More than 1 checkpoint available at {rc.checkpoints_path} ({ckpts_list})')
        ckpt_name = ckpts_list[-1]
    else:
        ckpt_name = f'epoch_{epoch:04d}.ckpt'
    return rc.checkpoints_path / ckpt_name


def load_model_checkpoint(rc, epoch='best'):
    checkpoint_path = get_checkpoint_path(rc, epoch=epoch)
    return rc.model_config.cls.load_from_checkpoint(checkpoint_path)


def load_rc_checkpoint(
    config,
    dataset_group,
    dataset,
    model=None,
    run_id=0,
    hparams=None,
    model_cls=None,
):
    # We suppose that, if hyperparameters are given, we want models that were obtained during
    # hyperparameter tuning with specific hyperparameters.
    tuning = hparams is not None
    rc = RunConfig(
        config=config,
        dataset_group=dataset_group,
        dataset=dataset,
        model=model,
        run_id=run_id,
        tuning=tuning,
        hparams=hparams,
        model_cls=model_cls,
    )
    if not rc.tuning:   # If the run was not obtained during hyperparameter tuning
        rc.hparams = get_hparams(config, rc)
    if model is not None:
        rc.model_config.args = OmegaConf.merge(rc.model_config.args, rc.hparams)
    return rc
