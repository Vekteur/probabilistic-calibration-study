import logging
from pathlib import Path

from ..base_datamodule import BaseDataModule
from .download_openml import download_openml_suite, load_dataset

log = logging.getLogger(__name__)


class InvalidDataset(Exception):
    pass


class OpenMLDataModule(BaseDataModule):
    def __init__(self, *args, suite_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.suite_id = suite_id

    def get_data(self):
        data_path = Path(self.rc.config.data_dir)
        path = data_path / 'openml' / str(self.suite_id) / self.hparams.name
        try:
            x, y = load_dataset(path)
        except FileNotFoundError:
            log.info('Downloading datasets...')
            download_openml_suite(self.suite_id, data_path)
            x, y = load_dataset(path)
        return x, y
