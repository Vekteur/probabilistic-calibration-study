from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def filter_dict(d, keys):
    return {key: d[key] for key in keys if key in d}


def inter(l1, l2):
    return [value for value in l1 if value in l2]


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end - start


def instantiate(config, *args, **kwargs):
    cfg_args = config['args'] if 'args' in config else {}
    return config.cls(*args, **cfg_args, **kwargs)


done_once = set()


def once(name):
    if name in done_once:
        return False
    done_once.add(name)
    return True


def print_once(name, string, box=True):
    if once(name):
        if box:
            horiz_line = '=' * (len(string) + 6)
            content = f'{horiz_line}\n|| {string} ||\n{horiz_line}'
            print(content, flush=True)
        else:
            print(string, flush=True)


def savefig(path, fig=None, **kwargs):
    if fig is None:
        fig = plt.gcf()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1)
    fig.savefig(
        path,
        bbox_extra_artists=fig.legends or None,
        bbox_inches='tight',
        **kwargs,
    )
    plt.close(fig)


def plot_or_savefig(path=None, fig=None, **kwargs):
    if path is None:
        plt.show()
    else:
        savefig(path, fig=fig, **kwargs)


def set_notebook_options():
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 80)
    pd.set_option('display.float_format', '{:.3f}'.format)
    # Avoid the SettingWithCopyWarning, which can be triggered even if there is no problem.
    # Just be extra careful when dealing with functions that can return a copy of a dataframe.
    pd.options.mode.chained_assignment = None
    mpl.rcParams['axes.formatter.limits'] = (-2, 4)
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    sns.set_theme()
