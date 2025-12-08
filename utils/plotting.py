# utils/plotting.py

import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt


def plot_monitor_csv(log_dir: str = "logs", monitor_file: str = "monitor.csv", window: int = 50, save_path: Optional[str] = None):
    path = os.path.join(log_dir, monitor_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} non trovato. Hai già lanciato il training?")

    df        = pd.read_csv(path, comment="#")
    rewards   = df["r"].values
    timesteps = df["l"].cumsum()

    if len(rewards) >= window:
        rolling = pd.Series(rewards).rolling(window=window).mean().values
    else:
        rolling = rewards

    plt.figure()
    plt.plot(timesteps, rewards, alpha=0.3, label="Episode Reward")
    plt.plot(timesteps, rolling, label=f"Moving avg ({window})")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.title("TD3 su Panda Door")
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()