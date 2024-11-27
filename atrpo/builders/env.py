import functools
from typing import Optional

import gym
import torch
from brax.envs.wrappers import torch as torch_wrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from atrpo.utils.env import BraxEnvWrapper, create_gym_env_on_specific_device


def create_gym_envs(env_id: str, n_envs: int, seed: Optional[int] = None) -> VecEnv:
    envs = make_vec_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
    )
    return envs


def create_brax_envs(
    env_id: str,
    env_name: str,
    n_envs: int,
    device: torch.device,
    gpu_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> BraxEnvWrapper:
    _ = torch.ones(1, device=device)

    entry_point = functools.partial(
        create_gym_env_on_specific_device, env_name=env_name
    )
    if env_id not in gym.envs.registry.env_specs:
        gym.register(env_id, entry_point=entry_point)

    # create a gym environment and wrap it
    gym_env = gym.make(env_id, batch_size=n_envs, gpu_id=gpu_id)
    gym_env = torch_wrapper.TorchWrapper(gym_env, device=device)

    # jit compile
    gym_env.reset()
    action = torch.rand(gym_env.action_space.shape, device=device) * 2 - 1
    gym_env.step(action)

    # wrap for compatibility with sb3
    envs = BraxEnvWrapper(
        env=gym_env,
        num_envs=n_envs,
    )
    envs = VecMonitor(envs)
    envs.seed(seed)
    return envs
