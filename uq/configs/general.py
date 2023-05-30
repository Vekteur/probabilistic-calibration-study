from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf


def get_log_dir(config):
    assert config.name not in ['fast', 'debug']
    optional_dirs = []
    for dir in ['fast', 'debug']:
        if config.get(dir):
            optional_dirs.append(dir)
    log_dir = Path(config.log_base_dir)
    if optional_dirs:
        log_dir /= '-'.join(optional_dirs)
    if config.name != 'unnamed':
        log_dir /= config.name
    else:
        log_dir /= datetime.now().strftime(r'%Y-%m-%d')
        log_dir /= datetime.now().strftime(r'%H-%M-%S')
    return log_dir


def general_config(config):
    work_dir = Path('.').resolve()
    default_config = OmegaConf.create(
        dict(
            work_dir=str(work_dir),
            data_dir=str(work_dir / 'data'),
            log_base_dir=str(work_dir / 'logs'),
            print_config=True,
            ignore_warnings=False,
            repeat_best=0,
            repeat_tuning=1,
            nb_workers=1,
            seed=0,
            name='unnamed',
            baseline_model='mixture_no_regul',
            default_batch_size=512,
            fast=False,
            debug=False,
            progress_bar=True,
            tuning=True,
            tuning_type='all',
            normalize=True,
            unnormalize=True,
            clean_previous=False,
            save_train_metrics=True,
            save_val_metrics=True,
            save_test_metrics=True,
            save_last_checkpoint=False,
            remove_checkpoints=False,
            # Modify the dataset using Y = mean(X) + epsilon where epsilon is known
            use_known_uncertainty=False,
        )
    )
    config = OmegaConf.merge(default_config, config)
    if config.fast:
        config.repeat_best = 1
    log_dir = get_log_dir(config)
    if config.name == 'unnamed' and log_dir.exists():
        raise RuntimeError('Unnamed experiment already exists')
    config.log_dir = str(log_dir)
    return config
