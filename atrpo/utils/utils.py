import math
import os
from typing import Callable

import hydra
import torch
from stable_baselines3.common.type_aliases import Schedule


def get_hydra_output_path(path: str) -> str:
    hydra_output_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    return os.path.join(hydra_output_path, path)


def get_smoothing_coeff_constant(const: float) -> Callable[[float, float], float]:
    def func(target_kl: float, curr_coeff: float) -> float:
        return const

    return func


def get_kl_update_ratio_exp_by_snr(
    kl_update_strength_func: Callable[[float, float], float],
    projection: Callable[[torch.Tensor], torch.Tensor],
    target_snr: float,
) -> Callable[[float, float, torch.Tensor], torch.Tensor]:
    def func(
        target_kl: float,
        smoothing_coeff: float,
        estimated_snr: torch.Tensor,
    ) -> torch.Tensor:
        kl_update_strength = kl_update_strength_func(target_kl, smoothing_coeff)
        exponent = projection(estimated_snr / (target_snr * target_kl) - 1)
        return torch.exp(kl_update_strength * exponent)

    return func


def get_kl_update_strength_const(const: float) -> Callable[[float, float], float]:
    def func(target_kl: float, smoothing_coeff: float) -> float:
        return const

    return func
