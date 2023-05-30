from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    TQDMProgressBar,
)


def callbacks_config(config):
    model_checkpoint_config = dict(
        cls=ModelCheckpoint,
        args=dict(
            mode='min',  # "max" means higher metric value is better, can be also "min"
            save_top_k=1,  # save k best models (determined by above metric)
            save_last=config.save_last_checkpoint,  # save model from last epoch
            verbose=False,
            dirpath='checkpoints/',
            filename='epoch_{epoch:04d}',
            auto_insert_metric_name=False,
        ),
    )
    early_stopping_config = dict(
        cls=EarlyStopping,
        args=dict(
            mode='min',  # "max" means higher metric value is better, can be also "min"
            patience=15,  # how many validation epochs of not improving until training stops
            min_delta=0,  # minimum change in the monitored metric needed to qualify as an improvement
        ),
    )
    progress_bar_config = dict(cls=TQDMProgressBar, args=dict(refresh_rate=10))
    callbacks_config = dict(
        model_checkpoint=model_checkpoint_config,
        early_stopping=early_stopping_config,
    )
    if config.progress_bar:
        callbacks_config['progress_bar'] = progress_bar_config
    return OmegaConf.create(dict(callbacks=callbacks_config), flags={'allow_objects': True})
