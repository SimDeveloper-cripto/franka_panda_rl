from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass

"""
    - Identified the cause of the 'struggling' behavior.
    - The w_action penalty (0.05) is high relative to the time_penalty (0.002), causing the optimal policy
        to use only ~20% force to minimize cost, resulting in slow/weak movement despite high success rate.
"""

@dataclass
class TrainConfig:
    seed   : int = 42
    run_dir: str = "runs/close_det"  # "runs/door_close_sac"
    tb_dir : str = "runs/tb"

    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 500
    control_freq: int = 30  # 20

    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = False  # Per imparare "chiudi + torna" deve proseguire dopo la chiusura

    num_envs    : int  = 8
    vecnormalize: bool = True

    total_steps    : int   = 3_000_000
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256
    gamma          : float = 0.99
    tau            : float = 0.005
    train_freq     : int   = 1
    gradient_steps : int   = 1
    learning_starts: int   = 20_000
    ent_coef       : str   = "auto_0.005"
    policy_net_arch: Tuple[int, int] = (256, 256)

    eval_freq      : int = 50_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 200_000

    close_fraction: float = 0.08

    init_open_min_fraction: float = 0.70
    init_open_max_fraction: float = 1.00

    w_progress   : float = 2.0
    w_delta      : float = 2.0

    # 1. Il robot non deve avere paura di spingere
    # 2. Deve avere più "fretta" di muoversi
    w_action     : float = 0.002  # was 0.001, was 0.05 way before
    time_penalty : float = 0.04   # was 0.002

    success_bonus    : float = 5.0
    debug_print_every: int   = 200

    # Post-success: return to start
    enable_return_stage: bool  = True
    w_return_pos       : float = 2.0
    w_door_regress     : float = 4.0
    return_pos_tol     : float = 0.05
    return_hold_steps  : int   = 10