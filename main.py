import os
import sys
from typing import List, Union

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

import wandb
from atrpo.utils.exp_manager import ExperimentManager
from atrpo.utils.mlflow_hydra import log_hydra_artifact, log_params_from_omegaconf


def experiment(
    config: DictConfig,
) -> Union[float, List[float]]:
    exp_manager = ExperimentManager(
        total_timesteps=config.experiment.total_timesteps,
        n_envs=config.experiment.n_envs,
        eval_freq=config.experiment.eval_freq,
        n_eval_envs=config.experiment.n_eval_envs,
        n_eval_episodes=config.experiment.n_eval_episodes,
        eval_deterministically=config.experiment.eval_deterministically,
        model_save_path=config.experiment.model_save_path,
        save_best_model=config.experiment.save_best_model,
        gpu_id_algo=config.experiment.gpu_id_algo,
        gpu_id_env=config.experiment.gpu_id_env,
        seed=config.experiment.seed,
        verbose=config.experiment.verbose,
    )

    model = exp_manager.setup_experiment(
        policy_config=config.policy,
        algo_config=config.algorithm,
        env_config=config.env,
        logger_config=config.logger,
    )
    exp_manager.learn(model)
    exp_manager.save_model(model)

    result = exp_manager.demo(
        model=model,
        n_episodes=config.demo.n_episodes,
        episode_length=config.demo.episode_length,
        env_config=config.env,
        deterministic=config.demo.deterministic,
        render_mode=config.demo.render_mode,
        video_config=config.logger.video,
        return_episode_rewards=config.demo.return_episode_rewards,
        record_suffix="_final_model",
    )
    if config.experiment.save_best_model:
        exp_manager.load_best_model()
        best_model_result = exp_manager.demo(
            model=model,
            n_episodes=config.demo.n_episodes,
            episode_length=config.demo.episode_length,
            env_config=config.env,
            deterministic=config.demo.deterministic,
            render_mode=config.demo.render_mode,
            video_config=config.logger.video,
            video_name_infix="-best",
            return_episode_rewards=config.demo.return_episode_rewards,
            record_suffix="_best_model",
        )
        if config.experiment.return_best_model_result:
            result = best_model_result

    if isinstance(result, list):
        result = [-r for r in result]
    else:
        result = -result

    return result


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    mlflow.set_tracking_uri("file://" + os.getcwd() + "/mlruns")
    mlflow.set_experiment(config.logger.experiment_name)
    if config.logger.wandb.use:
        wandb_config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True
        )
        run = wandb.init(
            project=config.logger.experiment_name,
            config=wandb_config,
            entity=config.logger.wandb.entity,
            group=config.logger.wandb.group,
            sync_tensorboard=True,
            save_code=True,
            reinit=True,
        )
    with mlflow.start_run():
        mlflow.set_tags(config.logger.mlflow.tags)
        log_params_from_omegaconf(config)
        log_hydra_artifact(sys.argv[0])
        result = experiment(config)
    if config.logger.wandb.use:
        run.finish()
    return result


if __name__ == "__main__":
    main()
