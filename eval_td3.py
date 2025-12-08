# eval_td3.py

import os
import numpy as np
from stable_baselines3 import TD3
from utils.make_env import make_single_env


def evaluate_model(model_path: str, n_episodes: int = 20):
    env   = make_single_env(seed=42, render=False)()
    model = TD3.load(model_path, env=env, print_system_info=True)

    rewards = []
    lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done      = False
        truncated = False
        ep_reward = 0.0
        ep_len    = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_len    += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)
        print(f"Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, len={ep_len}")

    print("\n================== Risultati ==================")
    print(f"Reward media: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Lunghezza media: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    env.close()


if __name__ == "__main__":
    model_path = os.path.join("models", "best_td3", "best_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join("models", "td3_panda_door_final.zip")
    evaluate_model(model_path)