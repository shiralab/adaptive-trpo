import os
from typing import Union

import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


def log_params_from_omegaconf(params: DictConfig):
    def _log_recursively(parent: str, child: Union[DictConfig, ListConfig]):
        if OmegaConf.is_dict(child):
            for key, value in child.items():
                if OmegaConf.is_dict(value) or OmegaConf.is_list(value):
                    _log_recursively(f"{parent}.{key}", value)
                else:
                    mlflow.log_param(f"{parent}.{key}", value)
        elif OmegaConf.is_list(child):
            for i, value in enumerate(child):
                mlflow.log_param(f"{parent}.{i}", value)

    for key, value in params.items():
        _log_recursively(key, value)


def log_hydra_artifact(filename: str):
    hydra_config = HydraConfig.get()
    output_dir = hydra_config.runtime.output_dir
    mlflow.log_artifact(os.path.join(output_dir, ".hydra/config.yaml"))
    mlflow.log_artifact(os.path.join(output_dir, ".hydra/hydra.yaml"))
    mlflow.log_artifact(os.path.join(output_dir, ".hydra/overrides.yaml"))
    basename_without_ext = os.path.splitext(os.path.basename(filename))[0]
    mlflow.log_artifact(os.path.join(output_dir, f"{basename_without_ext}.log"))
