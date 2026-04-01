from __future__ import annotations

from typing import Tuple
from dataclasses import dataclass


@dataclass
class TrainConfig:
    seed   : int = 42
    run_dir: str = "runs/close_det"
    tb_dir : str = "runs/tb"

    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 400
    control_freq: int = 30

    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = False

    num_envs    : int  = 8
    vecnormalize: bool = True

    total_steps    : int   = 1_000_000
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256

    # [FIX 9] gamma 0.99 → 0.95
    # Con gamma=0.99 il critic pesa quasi ugualmente reward a 100+ step di
    # distanza. La sequenza reach→grasp→close ha credito temporale lungo e
    # il critic fatica a propagarlo correttamente, causando oscillazioni
    # della success_rate. Con 0.95 il robot è più "miope" ma apprende la
    # causalità della sequenza più velocemente. Si può rialzare a 0.99
    # dopo che success_rate > 0.40 è stabile.
    gamma          : float = 0.95

    tau            : float = 0.005
    train_freq     : int   = 1
    gradient_steps : int   = 2
    learning_starts: int   = 5_000

    # floor a 0.05 per evitare collasso entropico (vedi log ent_coef=0.00015)
    # L'auto tuning nativo di SAC in Stable Baselines 3 è raccomandato:
    ent_coef       : str   = "auto"
    policy_net_arch: Tuple[int, int] = (256, 256)

    eval_freq      : int = 50_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 200_000

    close_fraction: float = 0.08

    # Task completo
    init_open_min_fraction: float = 0.70
    init_open_max_fraction: float = 1.00

    w_progress   : float = 2.0
    w_delta      : float = 2.0

    # [FIX 3] Azzerati: penalità azione e time_penalty gestite in env_gen.py
    # con logica per-fase. La classe madre non deve applicarle.
    w_action     : float = 0.0
    time_penalty : float = 0.0

    success_bonus    : float = 5.0
    debug_print_every: int   = 200

    enable_return_stage: bool  = False
    w_return_pos       : float = 2.0
    w_door_regress     : float = 4.0
    return_hold_steps  : int   = 10

    return_pos_tol      : float = 0.05
    action_smooth_alpha : float = 0.8

    limit_handle_friction : bool  = True
    handle_friction_max   : float = 0.8

    human_dist_min : float = 0.50
    human_dist_max : float = 0.60