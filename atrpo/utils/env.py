from typing import Any, List, Optional, Type, Union

import gym
import jax
import numpy as np
import torch
from brax.envs import create, wrappers
from brax.envs.base import PipelineEnv
from brax.envs.wrappers.gym import VectorGymWrapper
from brax.envs.wrappers.torch import TorchWrapper
from brax.io import image
from gym import spaces
from jax import numpy as jp
from jax.tree_util import tree_map
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def create_gym_env_on_specific_device(
    env_name: str,
    batch_size: Optional[int] = None,
    seed: int = 0,
    backend: Optional[str] = None,
    gpu_id: Optional[int] = None,
    **kwargs
) -> Union[gym.Env, gym.vector.VectorEnv]:
    """Creates a `gym.Env` or `gym.vector.VectorEnv` from a Brax environment."""
    environment = create(env_name=env_name, batch_size=batch_size, **kwargs)
    if batch_size is None:
        # TODO: support gpu_id
        return wrappers.GymWrapper(environment, seed=seed, backend=backend)
    if batch_size <= 0:
        raise ValueError("`batch_size` should either be None or a positive integer.")
    return VectorGymWrapperOnSpecificDevice(
        environment, seed=seed, backend=backend, gpu_id=gpu_id
    )


class VectorGymWrapperOnSpecificDevice(VectorGymWrapper):
    def __init__(
        self,
        env: PipelineEnv,
        seed: int = 0,
        backend: Optional[str] = None,
        gpu_id: Optional[int] = None,
    ) -> None:
        super().__init__(env, seed, backend)

        obs = np.inf * np.ones(env.observation_size, dtype="float32")
        self.single_obs_space = spaces.Box(-obs, obs, dtype="float32")
        action = jax.tree_map(np.array, env.sys.actuator.ctrl_range)
        self.single_action_space = spaces.Box(
            action[:, 0], action[:, 1], dtype="float32"
        )

        if gpu_id is None:
            return
        self.jax_device = jax.devices()[gpu_id]

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend, device=self.jax_device)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend, device=self.jax_device)

    def render(self, mode="human"):
        if mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("must call reset or step before rendering")
            state = tree_map(
                lambda x: jp.take(x, 0, axis=0, mode="wrap"), state.pipeline_state
            )
            return image.render_array(sys, state, 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception


class BraxEnvWrapper(VecEnv):
    def __init__(self, env: TorchWrapper, num_envs: int) -> None:
        super().__init__(num_envs, env.single_obs_space, env.single_action_space)
        self.env = env

    def reset(self) -> VecEnvObs:
        obs = self.env.reset()
        obs = obs.cpu().detach().numpy()
        return obs

    def step_async(self, actions) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, reward, done, infos_raw = self.env.step(torch.tensor(self.actions))
        obs = obs.cpu().detach().numpy()
        reward = reward.cpu().detach().numpy()
        done = done.cpu().detach().numpy()
        infos = []
        for i in range(self.num_envs):
            info = {}
            for key in infos_raw.keys():
                try:
                    # If info contains QP (with jax DeviceArray), this code will fail
                    # Maybe I should implement jax_to_torch() for QP...
                    info[key] = infos_raw[key][i]
                except TypeError:
                    pass
            infos.append(info)
        return obs, reward, done, infos

    def close(self) -> None:
        pass

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        return [getattr(self.env, attr_name)]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        return [getattr(self.env, method_name)(*method_args, **method_kwargs)]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(self.env, wrapper_class)]

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        return [self.env.seed(seed)]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        return self.env.render(mode)
