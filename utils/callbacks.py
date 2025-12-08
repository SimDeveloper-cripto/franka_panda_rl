# utils/callbacks.py

import os
from utils.make_env import make_single_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback


def create_eval_callback(log_dir: str = "logs", best_model_dir: str = "models/best_td3", eval_freq: int = 10_000, n_eval_episodes: int = 5, deterministic: bool = True, render: bool = False) -> EvalCallback:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    eval_env = DummyVecEnv([make_single_env(seed=123, render=render)])
    callback = EvalCallback(eval_env, best_model_save_path=best_model_dir, log_path=log_dir, eval_freq=eval_freq, n_eval_episodes=n_eval_episodes, deterministic=deterministic, render=render)
    return callback