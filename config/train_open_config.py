from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed   : int = 42  # 123
    run_dir: str = "runs/door_sac"
    tb_dir : str = "runs/tb"

    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 500
    control_freq: int = 20

    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = False

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
    learning_starts: int   = 10_000
    ent_coef       : str   = "auto"
    policy_net_arch: Tuple[int, int] = (256, 256)

    eval_freq      : int = 50_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 200_000

    success_fraction: float = 0.92

    w_progress   : float = 2.0
    w_delta      : float = 8.0
    w_action     : float = 0.02
    time_penalty : float = 0.002
    success_bonus: float = 5.0

    debug_print_every: int = 200

    w_action_post_success = 0.06

    # Post success: return to start
    enable_return_stage: bool = True
    w_return_pos       : float = 2.0
    w_door_regress     : float = 4.0
    return_pos_tol     : float = 0.05   # metri (tolleranza eef)
    return_hold_steps  : int   = 10

    action_deadband     : float = 0.02
    action_smooth_alpha : float = 0.2

    freeze_on_return : bool = True
    freeze_min_hold  : int  = 5