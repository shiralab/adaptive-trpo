import copy
import warnings
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, RolloutBufferSamples, Schedule
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class TRPOWithEnt(TRPO):
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
        smoothing_coeff: float = 0.01,
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
            cg_max_steps=cg_max_steps,
            cg_damping=cg_damping,
            line_search_shrinking_factor=line_search_shrinking_factor,
            line_search_max_iter=line_search_max_iter,
            n_critic_updates=n_critic_updates,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            normalize_advantage=normalize_advantage,
            target_kl=target_kl,
            sub_sampling_factor=sub_sampling_factor,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self.entcoeff = entcoeff
        self.smoothing_coeff = smoothing_coeff
        self._clip_smoothing_coeff()
        self.use_line_search = use_line_search

        policy_params = [
            param
            for name, param in self.policy.named_parameters()
            if "value" not in name
        ]
        num_params = sum(p.numel() for p in policy_params if p.requires_grad)
        self.cumulation_vg = th.zeros(num_params).to(self.device)
        self.cumulation_ng = th.zeros(num_params).to(self.device)
        self.cumulation_norm = 0.0
        self._dot_vg_ng = th.tensor(0.0).to(self.device)
        self.estimated_snr = th.tensor(0.0).to(self.device)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        This method is borrowed from sb3-contirb.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # Update target KL
        self._update_kl()
        self._update_smoothing_coeff()
        self._clip_smoothing_coeff()

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):
            # Optional: sub-sample data for faster computation
            if self.sub_sampling_factor > 1:
                rollout_data = RolloutBufferSamples(
                    rollout_data.observations[:: self.sub_sampling_factor],
                    rollout_data.actions[:: self.sub_sampling_factor],
                    None,  # old values, not used here
                    rollout_data.old_log_prob[:: self.sub_sampling_factor],
                    rollout_data.advantages[:: self.sub_sampling_factor],
                    rollout_data.returns[
                        :: self.sub_sampling_factor
                    ],  # None,  # returns, not used here
                )

            actions = rollout_data.actions
            if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                actions = rollout_data.actions.long().flatten()

            # Re-sample the noise matrix because the log_std has changed
            if self.use_sde:
                # batch_size is only used for the value function
                self.policy.reset_noise(actions.shape[0])

            with th.no_grad():
                # Note: is copy enough, no need for deepcopy?
                # If using gSDE and deepcopy, we need to use `old_distribution.distribution`
                # directly to avoid PyTorch errors.
                old_distribution = copy.copy(
                    self.policy.get_distribution(rollout_data.observations)
                )

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    rollout_data.advantages.std() + 1e-8
                )

            # ratio between old and new policy, should be one at the first iteration
            ratio = th.exp(log_prob - rollout_data.old_log_prob)

            # compute entropy
            entropy = distribution.entropy().mean()

            # surrogate policy objective
            policy_objective = (advantages * ratio).mean() + self.entcoeff * entropy

            # KL divergence
            kl_div = kl_divergence(distribution, old_distribution).mean()

            # Surrogate & KL gradient
            self.policy.optimizer.zero_grad()

            (
                actor_params,
                policy_objective_gradients,
                grad_kl,
                grad_shape,
            ) = self._compute_actor_grad(kl_div, policy_objective)

            # Hessian-vector dot product function used in the conjugate gradient step
            hessian_vector_product_fn = partial(
                self.hessian_vector_product, actor_params, grad_kl
            )

            # Computing search direction
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn,
                policy_objective_gradients,
                max_iter=self.cg_max_steps,
            )

            # Perform cumulation
            self._cumulate(
                policy_objective_gradients.detach().clone(),
                search_direction.detach().clone(),
            )

            # Maximal step length
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= th.matmul(
                search_direction,
                hessian_vector_product_fn(search_direction, retain_graph=False),
            )
            line_search_max_step_size = th.sqrt(line_search_max_step_size)

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            with th.no_grad():
                # Line-search (backtracking)
                for _ in range(self.line_search_max_iter):
                    step_size = line_search_backtrack_coeff * line_search_max_step_size
                    self._update_policy(
                        actor_params=actor_params,
                        original_actor_params=original_actor_params,
                        grad_shape=grad_shape,
                        step_size=step_size,
                        search_direction=search_direction,
                    )

                    # Recomputing the policy log-probabilities
                    distribution = self.policy.get_distribution(
                        rollout_data.observations
                    )
                    log_prob = distribution.log_prob(actions)

                    # New policy objective
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    entropy = distribution.entropy().mean()
                    new_policy_objective = (
                        advantages * ratio
                    ).mean() + self.entcoeff * entropy

                    # New KL-divergence
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    # Constraint criteria:
                    # we need to improve the surrogate policy objective
                    # while being close enough (in term of kl div) to the old policy
                    if (
                        (kl_div < self.target_kl)
                        and (new_policy_objective > policy_objective)
                        or not self.use_line_search
                    ):
                        is_line_search_success = True
                        break

                    # Reducing step size if line-search wasn't successful
                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if self.use_line_search and not is_line_search_success:
                    # If the line-search wasn't successful we revert to the original parameters
                    for param, original_param in zip(
                        actor_params, original_actor_params
                    ):
                        param.data = original_param.data.clone()

                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())

        # Critic update
        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                # Removing gradients of parameters shared with the actor
                # otherwise it defeats the purposes of the KL constraint
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        self._log_additional_info()

    def _update_policy(
        self,
        actor_params: List[th.nn.Parameter],
        original_actor_params: List[th.Tensor],
        grad_shape: List[Tuple[int, ...]],
        step_size: th.Tensor,
        search_direction: th.Tensor,
    ) -> None:
        start_idx = 0
        # Applying the scaled step direction
        for param, original_param, shape in zip(
            actor_params, original_actor_params, grad_shape
        ):
            n_params = param.numel()
            param.data = original_param.data + step_size * search_direction[
                start_idx : (start_idx + n_params)
            ].view(shape)
            start_idx += n_params

    def _update_kl(self) -> None:
        """Update target KL."""
        pass

    def _update_smoothing_coeff(self) -> None:
        """Update smoothing coefficient of cumulation."""
        pass

    def _clip_smoothing_coeff(self, eps: float = 1e-7) -> None:
        """Clip smoothing coefficient to avoid NaN."""
        self.smoothing_coeff = np.clip(self.smoothing_coeff, eps, 1.0 - eps)

    def _cumulate(self, vanilla_grad: th.Tensor, natural_grad: th.Tensor) -> None:
        """Perform cumulation."""
        norm_squared = th.dot(vanilla_grad, natural_grad)
        norm_vg = th.linalg.vector_norm(vanilla_grad)
        norm_ng = th.linalg.vector_norm(natural_grad)

        self.cumulation_vg = (1 - self.smoothing_coeff) * self.cumulation_vg + np.sqrt(
            self.smoothing_coeff * (2 - self.smoothing_coeff)
        ) * vanilla_grad / norm_vg
        self.cumulation_ng = (1 - self.smoothing_coeff) * self.cumulation_ng + np.sqrt(
            self.smoothing_coeff * (2 - self.smoothing_coeff)
        ) * natural_grad / norm_ng
        self.cumulation_norm = (
            1 - self.smoothing_coeff
        ) ** 2 * self.cumulation_norm + self.smoothing_coeff * (
            2 - self.smoothing_coeff
        ) * norm_squared / (
            norm_vg * norm_ng
        )

        self._dot_vg_ng = th.dot(self.cumulation_vg, self.cumulation_ng)
        self.estimated_snr = (
            (self._dot_vg_ng - self.cumulation_norm)
            * self.smoothing_coeff
            / (
                (2.0 - self.smoothing_coeff) * self.cumulation_norm
                - self.smoothing_coeff * self._dot_vg_ng
            )
        )

    def _log_additional_info(self) -> None:
        """Output additional logs."""
        self.logger.record("train/target_kl", self.target_kl)
        self.logger.record("train/smoothing_coeff", self.smoothing_coeff)
        self.logger.record("train/dot_vg_ng", self._dot_vg_ng.item())
        self.logger.record("train/cumulation_norm", self.cumulation_norm.item())
        self.logger.record(
            "train/s_over_gamma",
            (self._dot_vg_ng / self.cumulation_norm).item(),
        )
        self.logger.record("train/estimated_snr", self.estimated_snr.item())
