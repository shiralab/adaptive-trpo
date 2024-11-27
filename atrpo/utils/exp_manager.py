import os
import random
import sys
from typing import List, Optional, Union

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
)
from wandb.integration.sb3 import WandbCallback

import wandb
from atrpo.builders import build_algorithm, create_envs, get_policy_settings
from atrpo.utils.logger import MLflowOutputFormat
from atrpo.utils.utils import get_hydra_output_path


class ExperimentManager:
    """Manage experiment."""

    def __init__(
        self,
        total_timesteps: int,
        n_envs: int,
        eval_freq: int,
        n_eval_envs: int,
        n_eval_episodes: int,
        eval_deterministically: bool,
        model_save_path: str,
        save_best_model: bool,
        gpu_id_algo: int,
        gpu_id_env: int,
        seed: int,
        verbose: int,
    ) -> None:
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.eval_freq = eval_freq
        self.n_eval_envs = n_eval_envs
        self.n_eval_episodes = n_eval_episodes
        self.eval_deterministically = eval_deterministically
        self.model_save_path = model_save_path
        self.save_best_model = save_best_model
        self.gpu_id_algo = gpu_id_algo
        self.gpu_id_env = gpu_id_env
        self.seed = seed
        self.verbose = verbose

        self.env = None
        self.device_algo = None
        self.device_env = None
        self.loggers = None
        self.use_wandb = False
        self.callbacks = []

    def setup_experiment(
        self,
        policy_config: DictConfig,
        algo_config: DictConfig,
        env_config: DictConfig,
        logger_config: DictConfig,
    ) -> BaseAlgorithm:
        self.use_wandb = logger_config.wandb.use

        self._fix_seed(self.seed)
        self._set_device(
            gpu_id_algo=self.gpu_id_algo,
            gpu_id_env=self.gpu_id_env,
        )

        self.env = create_envs(
            n_envs=self.n_envs,
            device=self.device_env,
            config=env_config,
            seed=self.seed,
            gpu_id=self.gpu_id_env,
        )

        self.callbacks += self.create_callbacks(env_config)

        policy, policy_kwargs = get_policy_settings(policy_config)

        self.model = build_algorithm(
            policy=policy,
            env=self.env,
            config=algo_config,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            device=self.device_algo,
        )

        tensorboard_path = get_hydra_output_path(logger_config.tensorboard_path)
        self.loggers = Logger(
            folder=None,
            output_formats=[
                HumanOutputFormat(sys.stdout),
                MLflowOutputFormat(),
                TensorBoardOutputFormat(tensorboard_path),
            ],
        )
        self.model.set_logger(self.loggers)

        return self.model

    def _fix_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _set_device(self, gpu_id_algo: int, gpu_id_env: int) -> None:
        if torch.cuda.is_available():
            if gpu_id_algo >= 0:
                self.device_algo = torch.device(f"cuda:{gpu_id_algo}")
            else:
                self.device_algo = torch.device("cpu")

            if gpu_id_env >= 0:
                self.device_env = torch.device(f"cuda:{gpu_id_env}")
            else:
                self.device_env = torch.device("cpu")
        else:
            self.device_algo = torch.device("cpu")
            self.device_env = torch.device("cpu")

    def learn(self, model: BaseAlgorithm) -> None:
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=self.callbacks,
            )
        except KeyboardInterrupt:
            pass
        finally:
            self.loggers.dump(step=model.num_timesteps)
            try:
                model.env.close()
            except EOFError:
                pass

    def demo(
        self,
        model: BaseAlgorithm,
        n_episodes: int,
        episode_length: int,
        env_config: DictConfig,
        deterministic: bool = True,
        render_mode: Optional[str] = "rgb_array",
        video_config: Optional[DictConfig] = None,
        video_name_infix: str = "",
        return_episode_rewards: bool = False,
        record_suffix: str = "",
    ) -> Union[float, List[float]]:
        if render_mode != "rgb_array":
            video_config = None
        env = create_envs(
            n_envs=1,
            device=self.device_env,
            config=env_config,
            seed=self.seed,
            gpu_id=self.gpu_id_env,
            video_config=video_config,
            video_name_infix=video_name_infix,
        )

        episode_rewards = []

        for i in range(n_episodes):
            episode_reward = 0
            done = False
            obs = env.reset()
            for _ in range(episode_length):
                action, _state = model.predict(obs, deterministic=deterministic)
                obs, reward, done, _info = env.step(action)
                episode_reward += reward[0]

                if render_mode is not None and render_mode != "rgb_array":
                    env.render(mode=render_mode)

                if done:
                    break

            episode_rewards.append(episode_reward)
            self.loggers.record(f"demo/episode_reward{record_suffix}", episode_reward)
            self.loggers.dump(step=i)

        if video_config is not None:
            self._log_videos(video_config)

        env.close()

        if return_episode_rewards:
            return episode_rewards
        else:
            return np.mean(episode_rewards)

    def create_callbacks(self, env_config: DictConfig) -> list[BaseCallback]:
        callbacks = []
        if self.eval_freq > 0:
            eval_freq = max(self.eval_freq // self.n_envs, 1)
            eval_env = create_envs(
                n_envs=self.n_eval_envs,
                device=self.device_env,
                config=env_config,
                seed=self.seed,
                gpu_id=self.gpu_id_env,
            )
            best_model_save_path = None
            if self.save_best_model:
                best_model_save_path = get_hydra_output_path(self.model_save_path)
            eval_callback = EvalCallback(
                eval_env=eval_env,
                n_eval_episodes=self.n_eval_episodes,
                eval_freq=eval_freq,
                best_model_save_path=best_model_save_path,
                deterministic=self.eval_deterministically,
            )
            callbacks.append(eval_callback)
        if self.use_wandb:
            wandb_callback = WandbCallback()
            callbacks.append(wandb_callback)
        return callbacks

    def _log_videos(self, video_config: DictConfig) -> None:
        video_path = get_hydra_output_path(video_config.path)
        videos = [
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if os.path.splitext(f)[1] == ".mp4"
        ][: video_config.n_save_files]

        for video in videos:
            mlflow.log_artifact(video)

        if not self.use_wandb:
            return

        wandb.log({os.path.basename(video): wandb.Video(video) for video in videos})

    def save_model(self, model: BaseAlgorithm) -> None:
        """Save trained model to specified path."""
        save_path = get_hydra_output_path(self.model_save_path)
        model.save(os.path.join(save_path, "model"))
        print(f"Model saved to {save_path}")

    def load_model(self, model_path: str) -> None:
        """
        Load model from specified path.
        Assuming that `setup_experiment()` method has been already called.
        """
        self.model.set_parameters(
            load_path_or_dict=model_path,
            device=self.device_algo,
        )

    def load_best_model(self) -> None:
        """Load best model."""
        model_path = os.path.join(get_hydra_output_path(self.model_save_path), "best_model")
        self.load_model(model_path)
