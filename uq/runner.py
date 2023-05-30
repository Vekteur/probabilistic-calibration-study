import json
import logging
import pickle
import shutil
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dask.distributed import as_completed, get_client
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from uq.configs.tuning import get_tuning
from uq.train import train
from uq.utils.run_config import RunConfig

log = logging.getLogger('uq')


def get_best_hparams(results):
    df = pd.DataFrame(
        {
            'hparams': map(lambda rc: rc.hparams, results),
            # `hparams_id` is needed because groupby needs a hashable type
            'hparams_id': map(lambda rc: tuple(rc.hparams.items()), results),
            'score': map(lambda rc: rc.metrics['best_score'], results),
        }
    )
    assert len(df) > 0
    df = df.groupby('hparams_id').agg({'score': 'mean', 'hparams': lambda x: x.values[0]})
    best_score = df['score'].min()
    try:
        best_hparams = df.query(f'score == {best_score}')['hparams'].iloc[0]
    except KeyError:
        log.warn('The best score is nan')
        log.warn(df, flush=True)
        best_hparams = df['hparams'].iloc[0]
    return best_hparams


def mute_cumbersome_logging():
    logging.getLogger('lightning_lite.utilities.seed').setLevel(logging.WARNING)
    logging.getLogger('pytorch_lightning.utilities.rank_zero').setLevel(logging.WARNING)
    logging.getLogger('distributed.diskutils').setLevel(logging.WARN)
    warnings.filterwarnings('ignore', '.*Unmanaged memory use is high.*')
    warnings.filterwarnings(
        'ignore',
        '.*The `srun` command is available on your system but is not used.*',
        category=PossibleUserWarning,
    )
    warnings.filterwarnings(
        'ignore',
        '.*GPU available but not used. Set `accelerator` and `devices` using*',
        category=PossibleUserWarning,
    )


def train_and_save(rc, hparams, pm):
    logging.basicConfig(level=logging.WARN)
    mute_cumbersome_logging()

    rc.hparams = hparams
    if rc.storage_path.exists():
        with open(rc.storage_path, 'rb') as f:
            return pickle.load(f)

    if pm is not None:
        index = pm.request().result()
    else:
        index = 0
    rc = train(rc, index)
    if pm is not None:
        pm.free(index).result()

    rc.storage_path.parent.mkdir(parents=True, exist_ok=True)
    if rc.config.remove_checkpoints:
        assert len(list(rc.checkpoints_path.rglob('*'))) <= 2, list(rc.checkpoints_path.rglob('*'))
        # In case of error, check that I am not running the same runs (with same hyperparameters) concurrently!
        shutil.rmtree(rc.checkpoints_path)
    config = rc.config
    with open(rc.storage_path, 'wb') as f:
        # Don't save the whole config. This should save a lot of space.
        # However, it should be readded after loading the RunConfig again.
        rc.config = None
        pickle.dump(rc, f)
    rc.config = config
    return rc


class PositionManager:
    def __init__(self, size):
        self.slots = [False for _ in range(size)]

    def free(self, i):
        self.slots[i] = False

    def request(self):
        for i, slot in enumerate(self.slots):
            if not slot:
                self.slots[i] = True
                return i
        log.warn('No slot available')
        return 0


def submit(fn, *args, dask=True, priority=None, **kwargs):
    if dask:
        return get_client().submit(fn, *args, **kwargs, priority=priority)
    else:
        return fn(*args, **kwargs)


class Runner:
    def __init__(self, config, dask=True):
        self.config = config
        self.dask = dask
        self.tasks = []
        if self.dask:
            pm_future = submit(
                PositionManager,
                self.config.nb_workers,
                dask=self.dask,
                actor=True,
            )
            self.pm = pm_future.result()
        else:
            self.pm = None

    def train_in_parallel(self, rc, hparams, priority):
        return submit(
            train_and_save,
            rc,
            hparams,
            self.pm,
            dask=self.dask,
            priority=priority,
        )

    def grid_search(self, rc, priority):
        grid = get_tuning(rc.config)[rc.model]
        for hparams in grid:
            rc.model_cls = hparams.pop('model')
            future_rc = self.train_in_parallel(rc, hparams, priority)
            self.tasks.append(future_rc)

    def run_tuning(self, rc, priority):
        rc.tuning = True
        for run_id in range(self.config.repeat_tuning):
            rc.run_id = run_id
            self.grid_search(rc, priority)

    def close(self):
        if self.dask:
            for future in as_completed(self.tasks):
                if future.status == 'error':
                    log.warn('Error in parallel task')
                    print('=' * 60)
                    print('Traceback')
                    print('=' * 60)
                    traceback.print_tb(future.traceback())
                    print('Exception:', future.exception())


def run_all(config: DictConfig):
    logging.basicConfig(level=logging.WARN)
    log.setLevel(logging.WARN)
    OmegaConf.save(config, Path(config.log_dir) / 'config.yaml')

    runner = Runner(config, dask=config.nb_workers != 1)
    priority = 0
    for dataset_group, dataset_group_config in config.dataset_groups.items():
        # log.info(f"Dataset group: \033[1;4m{dataset_group}\033[0m")
        for dataset in dataset_group_config.names:
            # log.info(f"  Dataset: \033[4m{dataset}\033[0m")
            for model in get_tuning(config):
                # log.info(f"    Model: \033[1m{model}\033[0m")
                rc = RunConfig(
                    config=config,
                    dataset_group=dataset_group,
                    dataset=dataset,
                    model=model,
                )
                runner.run_tuning(rc, priority)
                priority -= 1
    runner.close()
