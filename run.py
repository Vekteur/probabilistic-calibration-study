import warnings
from pathlib import Path

from omegaconf import OmegaConf
from dask.distributed import Client, LocalCluster

# filter some warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Detected KeyboardInterrupt, attempting graceful shutdown....*")


def main():
    import sys

    from uq.configs.config import get_config
    from uq import utils
    from uq.runner import run_all

    config = OmegaConf.from_cli(sys.argv)
    config = get_config(config)
    config.clean_previous = True
    OmegaConf.resolve(config)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)
    # Dask client
    Client(n_workers=config.nb_workers, threads_per_worker=1, memory_limit=None)
    # Train model
    return run_all(config)


if __name__ == "__main__":
    main()
