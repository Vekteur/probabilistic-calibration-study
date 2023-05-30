from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from uq.models.default_trainer import DefaultTrainer


def lightning_trainer_config(config):
    # fast
    fast = OmegaConf.create()
    if config.fast:
        fast_args_config = dict(
            max_epochs=2,
            check_val_every_n_epoch=1,
            fast_dev_run=False,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            detect_anomaly=True,
        )
        fast = OmegaConf.create(dict(args=fast_args_config))

    # debug
    debug = OmegaConf.create()
    if config.debug:
        debug_args_config = dict(
            overfit_batches=0,
            track_grad_norm=-1,
            detect_anomaly=True,
        )
        debug = OmegaConf.create(dict(args=debug_args_config))

    # default
    default_config = dict(
        cls=Trainer,
        args=dict(
            accelerator='cpu',
            devices=1,
            min_epochs=1,
            max_epochs=5000,
            # number of validation steps to execute at the beginning of the training
            num_sanity_val_steps=0,
            log_every_n_steps=1,
            check_val_every_n_epoch=2,
            enable_model_summary=False,
            enable_progress_bar=config.progress_bar,
        ),
    )
    default = OmegaConf.create(default_config, flags={'allow_objects': True})
    return OmegaConf.merge(default, fast, debug)


def default_trainer_config(config):
    default_config = dict(cls=DefaultTrainer, args=dict())
    return OmegaConf.create(default_config, flags={'allow_objects': True})


def trainer_config(config):
    return OmegaConf.create(
        dict(
            trainer=dict(
                lightning=lightning_trainer_config(config),
                default=default_trainer_config(config),
            )
        )
    )
