# train_td3.py

import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from utils.make_env import make_single_env
from utils.callbacks import create_eval_callback


def main():
    log_dir   = "logs"
    model_dir = "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def make_monitored_env(seed: int = 0):
        env = make_single_env(seed=seed, render=False)()
        env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
        return env

    n_envs  = 4
    env_fns = [lambda i=i: make_monitored_env(seed=0 + i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    n_actions     = vec_env.action_space.shape[-1]
    action_noise  = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    policy_kwargs = dict(net_arch=[400, 300])

    model = TD3(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, buffer_size=1_000_000, learning_starts=10_000, batch_size=256,
        tau=0.005, gamma=0.98, train_freq=(1, "step"), gradient_steps=1,
        policy_kwargs=policy_kwargs, action_noise=action_noise,
        verbose=1,
    )

    eval_callback = create_eval_callback(
        log_dir=log_dir,
        best_model_dir=os.path.join(model_dir, "best_td3"),
        eval_freq=50_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, log_interval=10)

    model_path = os.path.join(model_dir, "td3_panda_door_final.zip")
    model.save(model_path)
    print(f"Modello salvato in: {model_path}")


if __name__ == "__main__":
    main()