# play_td3.py

import os
import time

from stable_baselines3 import TD3
from envs.panda_door_env import PandaDoorEnv


def main():
    env        = PandaDoorEnv(render_mode="human", horizon=250, reward_shaping=True, randomize_env=True)
    model_path = os.path.join("models", "best_td3", "best_model.zip")

    if not os.path.exists(model_path):
        model_path = os.path.join("models", "td3_panda_door_final.zip")

    model = TD3.load(model_path, env=env)

    obs, info = env.reset()
    done      = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(1.0 / env.metadata.get("render_fps", 20))
    env.close()


if __name__ == "__main__":
    main()