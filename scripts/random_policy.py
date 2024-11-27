import statistics

import gym
from hydra import compose, initialize
import numpy as np
import torch

from atrpo.builders import create_envs


ENVS = ["ant", "halfcheetah", "hopper", "inverted_double_pendulum", "inverted_pendulum", "pusher", "reacher", "swimmer", "walker2d"]
NUM_EPISODES = 100
EPISODE_LENGTH = 1000
GPU_ID = 0
SEED = 100
FILENAME = "random_policy_reward.csv"


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id >= 0:
            device = torch.device(f"cuda:{gpu_id}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    return device


def execute_random_policy(env_cfg, device, render=False):
    env = create_envs(
        n_envs=1,
        device=device,
        config=env_cfg,
        seed=SEED,
        gpu_id=GPU_ID,
    )

    episode_rewards = []

    for i in range(NUM_EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()
        for _ in range(EPISODE_LENGTH):
            action = np.array([env.action_space.sample()])
            obs, reward, done, _info = env.step(action)
            episode_reward += reward[0]

            if render:
                env.render()

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"{env_cfg.name}-{i}:{episode_reward}")

    return statistics.median(episode_rewards)


def main():
    device = get_device(gpu_id=GPU_ID)
    with open(FILENAME, mode="x") as f:
        f.write("env,reward\n")
    initialize(version_base=None, config_path="../conf/env")
    for env in ENVS:
        cfg = compose(config_name=env)
        ret = execute_random_policy(env_cfg=cfg, device=device)
        with open(FILENAME, mode="a") as f:
            f.write(f"{env},{ret}\n")


if __name__ == "__main__":
    main()
