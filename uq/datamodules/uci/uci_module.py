import logging
from pathlib import Path

from ..base_datamodule import BaseDataModule
from .download_uci import download_all_uci, load_dataset

log = logging.getLogger(__name__)


class UCIDataModule(BaseDataModule):
    def get_data(self):
        data_path = Path(self.rc.config.data_dir)
        path = data_path / 'uci' / self.hparams.name
        try:
            x, y = load_dataset(path)
        except FileNotFoundError:
            log.info('Downloading datasets...')
            download_all_uci(data_path)
            x, y = load_dataset(path)
        return x, y
