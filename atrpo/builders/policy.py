from typing import Any, Tuple, Type

import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn


def prepare_kwargs_actor_critic(
    config: DictConfig,
) -> Tuple[Type[BaseModel], dict[str, Any]]:
    type = ActorCriticPolicy

    if config.activation_fn == "tanh":
        activation_fn = nn.Tanh
    else:
        raise NotImplementedError("Unknown activation function.")

    if config.feature_extractor.name == "flatten-extractor":
        features_extractor_class = FlattenExtractor
        features_extractor_kwargs = OmegaConf.to_object(config.feature_extractor)
        assert isinstance(features_extractor_kwargs, dict)
        features_extractor_kwargs.pop("name")
    else:
        raise NotImplementedError("Unknown feature extractor.")

    if config.optimizer.name == "adam":
        optimizer_class = torch.optim.Adam
        optimizer_kwargs = OmegaConf.to_object(config.optimizer)
        assert isinstance(optimizer_kwargs, dict)
        optimizer_kwargs.pop("name")
    else:
        raise NotImplementedError("Unknown optimizer.")

    kwargs = {
        "net_arch": OmegaConf.to_object(config.net_arch),
        "activation_fn": activation_fn,
        "ortho_init": config.ortho_init,
        "log_std_init": config.log_std_init,
        "full_std": config.full_std,
        "squash_output": config.squash_output,
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
        "normalize_images": config.normalize_images,
        "optimizer_class": optimizer_class,
        "optimizer_kwargs": optimizer_kwargs,
    }

    return type, kwargs
