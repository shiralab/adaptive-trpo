import os
from typing import Any, Optional, Tuple, Type, Union

import torch
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BaseModel
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv, VecVideoRecorder

from atrpo.builders.algorithm import (
    build_adaptive_trpo,
    build_ppo,
    build_trpo,
)
from atrpo.builders.env import create_brax_envs, create_gym_envs
from atrpo.builders.policy import prepare_kwargs_actor_critic
from atrpo.utils.utils import get_hydra_output_path


def build_algorithm(
    policy: Union[str, Type[BaseModel]],
    env: Union[str, GymEnv],
    config: DictConfig,
    **kwargs,
) -> BaseAlgorithm:
    if config.name in ("trpo", "npg"):
        assert isinstance(policy, str) or issubclass(policy, ActorCriticPolicy)
        return build_trpo(policy, env, config, **kwargs)
    elif config.name == "adaptive-trpo":
        assert isinstance(policy, str) or issubclass(policy, ActorCriticPolicy)
        return build_adaptive_trpo(policy, env, config, **kwargs)
    else:
        raise NotImplementedError(f"Unknown algorithm: {config.name}")


def get_policy_settings(config: DictConfig) -> Tuple[Type[BaseModel], dict[str, Any]]:
    if config.name == "actor-critic":
        return prepare_kwargs_actor_critic(config)
    else:
        raise NotImplementedError(f"Unknown policy: {config.name}")


def create_envs(
    n_envs: int,
    device: torch.device,
    config: DictConfig,
    seed: Optional[int] = None,
    gpu_id: Optional[int] = None,
    video_config: Optional[DictConfig] = None,
    video_name_infix: str = "",
) -> VecEnv:
    env = None
    if config.backend == "Gym":
        env = create_gym_envs(
            env_id=config.env_id,
            n_envs=n_envs,
            seed=seed,
        )
    elif config.backend == "Brax":
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env = create_brax_envs(
            env_id=config.env_id,
            env_name=config.name,
            n_envs=n_envs,
            device=device,
            gpu_id=gpu_id,
            seed=seed,
        )
    else:
        raise NotImplementedError(f"Unknown backend: {config.backend}")

    if video_config is not None:
        env.render_mode = "rgb_array"
        video_path = get_hydra_output_path(video_config.path)
        env = VecVideoRecorder(
            venv=env,
            video_folder=video_path,
            record_video_trigger=lambda _: False,
            video_length=video_config.length,
            name_prefix=f"{video_config.name_prefix}{video_name_infix}",
        )

    return env
