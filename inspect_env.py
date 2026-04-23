import os

import sys
sys.path.append('') # here insert path to workspace folder

from config.train_close_config import TrainConfig
from train_close import RoboSuiteDoorCloseGymnasiumEnv

cfg = TrainConfig()
env = RoboSuiteDoorCloseGymnasiumEnv(cfg)

obs_dict = env._rs_env.reset()

print("--- OBSERVATIONS ---")
for k, v in obs_dict.items():
    if hasattr(v, 'shape'):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)}")

print("\n--- ACTION SPACE ---")
print(env.action_space)
print("Action space low:" , env.action_space.low)
print("Action space high:", env.action_space.high)