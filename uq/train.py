import logging
import shutil
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.plugins.environments import SLURMEnvironment

from uq.utils.general import instantiate

log = logging.getLogger('uq')


class KeyboardInterruptCallback(Callback):
    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            exit()


class DisableLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def __enter__(self):
        self.logger.propagate = False

    def __exit__(self, type, value, traceback):
        self.logger.propagate = True


def instantiate_callback(rc, cb_conf, monitor, process_position):
    if cb_conf.cls == TQDMProgressBar:
        cb_conf.args.process_position = process_position
    if cb_conf.cls in [EarlyStopping, ModelCheckpoint]:
        cb_conf.args.monitor = f'val/{monitor}'
    callback = instantiate(cb_conf)
    if cb_conf.cls == ModelCheckpoint:
        callback.dirpath = str(rc.checkpoints_path)
    return callback


def init_callbacks(rc, monitor, process_position):
    checkpoints_path = rc.checkpoints_path

    # Checkpoints in this directory must originate from a previous interrupted run.
    # They should be cleaned up.
    if rc.config.clean_previous:
        if checkpoints_path.exists():
            shutil.rmtree(checkpoints_path)
    callbacks = []
    if 'callbacks' in rc.config:
        for cb_conf in rc.config.callbacks.values():
            if cb_conf is None:
                continue
            callback = instantiate_callback(rc, cb_conf, monitor, process_position)
            if callback is not None:
                callbacks.append(callback)
    callbacks.append(KeyboardInterruptCallback())
    return callbacks


def init_loggers(rc):
    loggers = []
    if 'loggers' in rc.config:
        for lg_conf in rc.config.loggers.values():
            loggers.append(instantiate(lg_conf))
    return loggers


def init_trainer(rc, trainer_name, callbacks, loggers):
    with DisableLogger('pytorch_lightning.utilities.distributed'):
        trainer = instantiate(
            rc.config.trainer[trainer_name],
            callbacks=callbacks,
            logger=loggers,
            plugins=[SLURMEnvironment(auto_requeue=False)],
        )
    return trainer


def init_trainer_with_loggers_and_callbacks(rc, model, trainer_name, process_position=0):
    # We set process_position at 0 for now because it is hard to setup with multiprocessing
    callbacks = init_callbacks(rc, model.monitor, process_position=process_position)
    loggers = init_loggers(rc)
    return init_trainer(rc, trainer_name, callbacks, loggers)


def get_model_args(rc, datamodule):
    datamodule.load_datasets()
    first_x, first_y = datamodule.data_train[0]
    input_size = first_x.shape[0]
    args = dict(rc.model_config.args)
    args.update(
        dict(
            rc=rc,
            input_size=input_size,
        )
    )
    args.update(rc.hparams)
    return args


def get_model(rc, datamodule):
    model_args = get_model_args(rc, datamodule)
    return rc.model_config.cls(**model_args)


def load_datamodule(rc):
    try:
        datamodule_config = rc.dataset_group_config.datamodule
    except ConfigAttributeError:
        log.error(f'The datamodule of {rc.dataset} is not present in the config')
        raise
    return instantiate(datamodule_config, rc=rc, name=rc.dataset, seed=2000 + rc.run_id)


def fit_with_profiling(rc, trainer, model, datamodule):
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        use_cuda=False,
    ) as prof:
        with record_function('training'):
            trainer.fit(model=model, datamodule=datamodule)
    prof.export_chrome_trace('tmp/trace.json')
    print(f'Profiling of {rc.summary_str()}', flush=True)
    print(
        prof.key_averages().table(sort_by='cpu_time_total', row_limit=50),
        flush=True,
    )


def train(rc, process_index):
    log.warn(f'Starting {rc.summary_str()}')
    # Init lightning datamodule
    # For each run index, different splits are selected.
    # An alternative would be to remove the randomness of the splits by fixing the same seed for each run index.
    datamodule = load_datamodule(rc)

    # Init lightning model
    # Use a random seed for the model
    if rc.tuning:
        seed = 1000 + rc.run_id
    else:
        seed = rc.run_id
    rc.seed = seed
    seed_everything(rc.seed)
    model = get_model(rc, datamodule)
    trainer = init_trainer_with_loggers_and_callbacks(
        rc,
        model,
        trainer_name=rc.model_config.trainer,
        process_position=process_index,
    )

    # Training loop
    trainer.fit(model=model, datamodule=datamodule)

    # Test the model
    for cb in trainer.callbacks:
        if type(cb) == ModelCheckpoint:
            assert cb.best_model_path != ''
    with DisableLogger('pytorch_lightning.utilities.distributed'):
        trainer.test(model=model, datamodule=datamodule, ckpt_path='best', verbose=False)
        # We compute the validation on the best epoch in all cases because it is used for model selection
        if not rc.config.save_val_metrics:
            rc.config.save_val_metrics = True
            trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path='best',
                verbose=False,
            )
            rc.config.save_val_metrics = False

    # Save the metrics
    # Convert any defaultdict to dict due to a bug (https://github.com/python/cpython/issues/79721)
    rc.metrics = dict(model.metrics_collector.metrics)
    log.warn(f'Finished {rc.summary_str()}')
    return rc
