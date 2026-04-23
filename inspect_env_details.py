import os

import sys
sys.path.append('') # here insert path to workspace folder

import numpy as np
from config.train_close_config import TrainConfig
from train_close import RoboSuiteDoorCloseGymnasiumEnv

cfg = TrainConfig()
env = RoboSuiteDoorCloseGymnasiumEnv(cfg)

obs_dict = env._rs_env.reset()

print("--- RAW OBSERVATIONS FROM ROBOSUITE ---")
for k, v in obs_dict.items():
    if hasattr(v, 'shape'):
        print(f"{k}: {v.shape} (ndim: {getattr(v, 'ndim', 'None')})")
    else:
        print(f"{k}: {type(v)}")

print("\n--- OBSERVATIONS FLATTENED FOR NEURAL NETWORK ---")
keys = sorted(k for k, v in obs_dict.items() if isinstance(v, np.ndarray) and v.ndim == 1)
print(f"Keys used: {keys}")

flat = np.concatenate([obs_dict[k].astype(np.float32).ravel() for k in keys])
print(f"Total input size to NN: {flat.shape[0]}")

print("\n--- ACTION SPACE ---")
print(env.action_space)
print("Action space low:" , env.action_space.low)
print("Action space high:", env.action_space.high)