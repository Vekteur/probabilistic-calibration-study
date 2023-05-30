import time

from uq import utils
from uq.utils.checkpoints import load_model_checkpoint, load_rc_checkpoint

log = utils.get_logger(__name__)


def build_baseline_model(rc):
    while True:
        try:
            baseline_rc = load_rc_checkpoint(
                config=rc.config,
                dataset_group=rc.dataset_group,
                dataset=rc.dataset,
                model=rc.config.baseline_model,
            )
            baseline_model = load_model_checkpoint(baseline_rc)
        except FileNotFoundError:
            wait_seconds = 20
            log.warn(f'Baseline model not found, waiting {wait_seconds} seconds')
            time.sleep(wait_seconds)
        else:
            break
    return baseline_model
