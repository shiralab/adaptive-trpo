from typing import Any, Callable, Optional, Type, Union

import torch
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import constant_fn, get_device

from atrpo.algorithms.adaptive_trpo import AdaptiveTRPO
from atrpo.algorithms.trpo_with_ent import TRPOWithEnt
from atrpo.utils.utils import (get_kl_update_ratio_exp_by_snr,
                               get_kl_update_strength_const,
                               get_smoothing_coeff_constant)


def _build_schedule(config: DictConfig) -> Schedule:
    if config.name == "constant":
        schedule = constant_fn(config.value)
    else:
        raise NotImplementedError(f"Unknown schedule: {config.name}")

    return schedule


def _build_projection(config: DictConfig) -> Callable[[torch.Tensor], torch.Tensor]:
    if config.name == "clamp":
        projection = lambda x: torch.clamp(x, config.min, config.max)
    else:
        raise NotImplementedError(f"Unknown projection: {config.name}")
    return projection


def _build_smoothing_coeff_func(config: DictConfig) -> Callable[[float, float], float]:
    if config.name == "constant":
        smoothing_coeff_func = get_smoothing_coeff_constant(const=config.const)
    else:
        raise NotImplementedError(f"Unknown smoothing coeff func: {config.name}")

    return smoothing_coeff_func


def _build_kl_update_ratio_func(
    config: DictConfig,
) -> Union[
    Callable[[float, float, torch.Tensor, torch.Tensor], torch.Tensor],
    Callable[[float, float, torch.Tensor], torch.Tensor],
]:
    if config.kl_update_strength_func.name == "constant":
        kl_update_strength_func = get_kl_update_strength_const(
            const=config.kl_update_strength_func.const
        )
    else:
        raise NotImplementedError(
            f"Unknown KL update strength func: {config.kl_update_strength_func.name}"
        )

    if config.name == "exp-by-snr":
        projection = _build_projection(config.projection)
        kl_update_ratio_func = get_kl_update_ratio_exp_by_snr(
            kl_update_strength_func=kl_update_strength_func,
            projection=projection,
            target_snr=config.target_snr,
        )
    else:
        raise NotImplementedError(f"Unknown KL update ratio func: {config.name}")

    return kl_update_ratio_func


def build_trpo(
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[str, GymEnv],
    config: DictConfig,
    policy_kwargs: Optional[dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "auto",
    init_setup_model: bool = True,
) -> TRPOWithEnt:
    lr_schedule = _build_schedule(config.learning_rate)
    return TRPOWithEnt(
        policy=policy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        entcoeff=config.entcoeff,
        cg_max_steps=config.cg_max_steps,
        cg_damping=config.cg_damping,
        use_line_search=config.use_line_search,
        line_search_shrinking_factor=config.line_search_shrinking_factor,
        line_search_max_iter=config.line_search_max_iter,
        n_critic_updates=config.n_critic_updates,
        gae_lambda=config.gae_lambda,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        normalize_advantage=config.normalize_advantage,
        target_kl=config.target_kl,
        smoothing_coeff=config.smoothing_coeff,
        sub_sampling_factor=config.sub_sampling_factor,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
        _init_setup_model=init_setup_model,
    )


def build_adaptive_trpo(
    policy: Union[str, Type[ActorCriticPolicy]],
    env: Union[str, GymEnv],
    config: DictConfig,
    policy_kwargs: Optional[dict[str, Any]] = None,
    verbose: int = 0,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "auto",
    init_setup_model: bool = True,
) -> AdaptiveTRPO:
    lr_schedule = _build_schedule(config.learning_rate)
    smoothing_coeff_func = _build_smoothing_coeff_func(config.smoothing_coeff_func)
    kl_update_ratio_func = _build_kl_update_ratio_func(config.kl_update_ratio_func)

    return AdaptiveTRPO(
        policy=policy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        entcoeff=config.entcoeff,
        cg_max_steps=config.cg_max_steps,
        cg_damping=config.cg_damping,
        use_line_search=config.use_line_search,
        line_search_shrinking_factor=config.line_search_shrinking_factor,
        line_search_max_iter=config.line_search_max_iter,
        n_critic_updates=config.n_critic_updates,
        gae_lambda=config.gae_lambda,
        use_sde=config.use_sde,
        sde_sample_freq=config.sde_sample_freq,
        normalize_advantage=config.normalize_advantage,
        target_kl=config.target_kl,
        kl_range=config.kl_range,
        init_smoothing_coeff=config.init_smoothing_coeff,
        smoothing_coeff_func=smoothing_coeff_func,
        kl_update_ratio_func=kl_update_ratio_func,
        sub_sampling_factor=config.sub_sampling_factor,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
        _init_setup_model=init_setup_model,
    )
