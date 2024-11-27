from copy import deepcopy
from functools import partial
from multiprocessing import Manager
from queue import Queue
from typing import List, Union

import hydra
import numpy as np

from hydra import initialize
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from scipy.stats import trim_mean
from torch.multiprocessing import Pool, set_start_method

from main import main

try:
    set_start_method("spawn")
except RuntimeError:
    pass


def single_run(
    config: DictConfig, hydra_cfg: HydraConf, queue: Queue, seed: int
) -> Union[float, List[float]]:
    gpu_id = -1
    if config.hpo.device_env == "gpu" or config.hpo.device_algo == "gpu":
        gpu_id = queue.get()

    with initialize(version_base=None, config_path="conf"):
        HydraConfig.instance().set_config(OmegaConf.create({"hydra": hydra_cfg}))

        run_cfg = deepcopy(config)

        run_cfg.experiment.seed = seed
        run_cfg.logger.tensorboard_path += f"-{seed}"
        run_cfg.logger.video.path += f"-{seed}"

        if config.hpo.device_env == "gpu":
            run_cfg.experiment.gpu_id_env = gpu_id
        if config.hpo.device_algo == "gpu":
            run_cfg.experiment.gpu_id_algo = gpu_id

        result = main(run_cfg)

    if gpu_id != -1:
        queue.put(gpu_id)

    return result


def aggregate(
    arr: Union[List[float], List[List[float]]], aggregator: str
) -> Union[float, List[float]]:
    arr = np.array(arr)

    if aggregator == "mean":
        ret = np.mean(arr, axis=0)
    elif aggregator == "median":
        ret = np.median(arr, axis=0)
    elif aggregator == "iqm":
        ret = trim_mean(arr, proportiontocut=0.25, axis=0)
    else:
        raise NotImplementedError(f"Unknown aggregator: {aggregator}")

    return ret.tolist()


@hydra.main(version_base=None, config_path="conf", config_name="config_ax")
def hpo(config: DictConfig):
    queue = Manager().Queue()
    for gpu_id in config.hpo.available_gpu_ids:
        queue.put(gpu_id)

    hydra_cfg = HydraConfig.get()

    results = []
    with Pool(processes=config.hpo.num_processes) as pool:
        for result in pool.imap_unordered(
            partial(single_run, config, hydra_cfg, queue), config.hpo.seeds
        ):
            results.append(result)
    return aggregate(results, config.hpo.aggregator)


if __name__ == "__main__":
    hpo()
