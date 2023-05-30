import logging
from collections import defaultdict
from dataclasses import asdict

import pandas as pd

log = logging.getLogger('uq')


class RunsStorage:
    """This class is used to store statistics about the runs.
    It is then transformed to a pandas dataframe to make queries efficiently.
    We don't use a dataframe directly because it is slow to insert to.
    It is a dictionary of lists of the same size.
    """

    def __init__(self):
        self.data = defaultdict(list)

    def add(self, rc):
        mode = 'tuning' if rc.tuning else 'best'
        log.debug(
            f'Added Run: \033[1m{rc.run_id}\033[0m, mode: {mode}, model: {rc.model}, hparams: {rc.hparams_str()}'
        )
        d = rc.__dict__
        assert len(self.data) == 0 or set(self.data) == set(d)
        for key, value in d.items():
            self.data[key].append(value)

    def build_df(self):
        return pd.DataFrame(self.data)

    def __len__(self):
        if len(self.data) == 0:
            return 0
        return len(next(iter(self.data.values())))

    # Search in O(n), be careful not to abuse it
    def __contains__(self, d):
        for i in range(len(self)):
            for key, value in d.items():
                if self.data[key][i] != value:
                    break
            else:
                return True
        return False
