import sys
from typing import Any, Callable, Optional, Type, Union

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from atrpo.algorithms.trpo_with_ent import TRPOWithEnt


class AdaptiveTRPO(TRPOWithEnt):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        n_steps: int = 2048,
        batch_size: int = 128,
        gamma: float = 0.99,
        entcoeff: float = 0.0,
        cg_max_steps: int = 15,
        cg_damping: float = 0.1,
        use_line_search: bool = True,
        line_search_shrinking_factor: float = 0.8,
        line_search_max_iter: int = 10,
        n_critic_updates: int = 10,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = True,
        target_kl: float = 0.01,
        kl_range: tuple[float, float] = (1e-9, float("inf")),
        init_smoothing_coeff: float = 0.01,
        smoothing_coeff_func: Callable[[float, float], float] = (lambda x, _: x),
        kl_update_ratio_func: Callable[
            [float, float, th.Tensor, th.Tensor], th.Tensor
        ] = (lambda *x: x[-1]),
        sub_sampling_factor: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            entcoeff=entcoeff,
            cg_max_steps=cg_max_steps,
            cg_damping=cg_damping,
            use_line_search=use_line_search,
            line_search_shrinking_factor=line_search_shrinking_factor,
            line_search_max_iter=line_search_max_iter,
            n_critic_updates=n_critic_updates,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            normalize_advantage=normalize_advantage,
            target_kl=target_kl,
            smoothing_coeff=init_smoothing_coeff,
            sub_sampling_factor=sub_sampling_factor,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.kl_range = kl_range
        self.smoothing_coeff_func = smoothing_coeff_func
        self.kl_update_ratio_func = kl_update_ratio_func

        policy_params = [
            param
            for name, param in self.policy.named_parameters()
            if "value" not in name
        ]
        num_params = sum(p.numel() for p in policy_params if p.requires_grad)
        self.cumulation_ng_elem = th.zeros(num_params).to(self.device)
        self.cumulation_one_elem = th.zeros(num_params).to(self.device)

    def _update_kl(self) -> None:
        self.target_kl = self.target_kl * self.kl_update_ratio_func(
            self.target_kl, self.smoothing_coeff, self.estimated_snr
        )

        self.target_kl = th.clamp(
            self.target_kl, min=self.kl_range[0], max=self.kl_range[1]
        ).item()

    def _update_smoothing_coeff(self) -> None:
        self.smoothing_coeff = self.smoothing_coeff_func(
            self.target_kl, self.smoothing_coeff
        )
