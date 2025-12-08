# utils/make_env.py

from typing import Callable

import gymnasium as gym
from envs.panda_door_env import PandaDoorEnv
from stable_baselines3.common.vec_env import DummyVecEnv


def make_single_env(seed: int = 0, render: bool = False):
    def _init():
        render_mode = "human" if render else None
        env = PandaDoorEnv(render_mode=render_mode, seed=seed, horizon=250, reward_shaping=True, randomize_env=True)
        return env
    return _init


def make_vec_env(n_envs: int = 1, seed: int = 0) -> gym.Env:
    env_fns: list[Callable] = [
        make_single_env(seed=seed + i, render=False) for i in range(n_envs)
    ]
    return DummyVecEnv(env_fns)