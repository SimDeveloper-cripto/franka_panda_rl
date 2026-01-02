# train_curriculum.py

from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, List

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from env_goal_door import GoalDoorEnv
from teacher import StageTeacher, StageSpec
from config.train_open_config import TrainConfig


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def save_config(cfg: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

def make_env_fn(cfg: TrainConfig, teacher: StageTeacher, render_mode: Optional[str] = None):
    def _thunk():
        return GoalDoorEnv(cfg=cfg, teacher=teacher, render_mode=render_mode, door_open_cap_rad=0.400)
    return _thunk

class TeacherUpdateCallback(BaseCallback):
    def __init__(self, teacher: StageTeacher, verbose: int = 0):
        super().__init__(verbose)
        self.teacher   = teacher
        self._ep_count = 0

    def _on_step(self) -> bool:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones                       = self.locals.get("dones", None)

        if dones is None:
            return True

        # For each environment instance, when (done == True), we treat it as episode end
        for done, info in zip(dones, infos):
            if not done:
                continue

            # Success for curriculum: use info["is_success"] if present, else infer from episode data
            success = bool(info.get("is_success", False))
            self.teacher.update(success=success)
            self._ep_count += 1

        if self._ep_count > 0 and (self._ep_count % 100 == 0):
            st = self.teacher.stats()
            self.logger.record("curriculum/stage_idx", st["stage_idx"])
            self.logger.record("curriculum/success_rate_window", st["success_rate_window"])

        return True


def build_teacher(seed: int) -> StageTeacher:
    # Stage curriculum tuned for limit (0.400 rad cap).
    stages = (
        StageSpec(
            name="S0_easy",
            goal_frac_min=0.05, goal_frac_max=0.20,
            friction_scale_min=1.0, friction_scale_max=1.0,
            damping_scale_min=1.0,  damping_scale_max=1.0,
        ),
        StageSpec(
            name="S1_mid",
            goal_frac_min=0.20, goal_frac_max=0.50,
            friction_scale_min=0.9, friction_scale_max=1.1,
            damping_scale_min=0.9,  damping_scale_max=1.2,
        ),
        StageSpec(
            name="S2_harder",
            goal_frac_min=0.50, goal_frac_max=0.80,
            friction_scale_min=0.7, friction_scale_max=1.3,
            damping_scale_min=0.8,  damping_scale_max=1.6,
        ),
        StageSpec(
            name="S3_full",
            goal_frac_min=0.80, goal_frac_max=1.00,
            friction_scale_min=0.6, friction_scale_max=1.5,
            damping_scale_min=0.7,  damping_scale_max=2.0,
        ),
        StageSpec(
            name="S4_mix",
            goal_frac_min=0.05, goal_frac_max=1.00,
            friction_scale_min=0.5, friction_scale_max=1.5,
            damping_scale_min=0.5,  damping_scale_max=2.0,
        ),
    )

    return StageTeacher(
        stages            = stages,
        window_episodes   = 200,
        promote_threshold = 0.85,
        seed              = seed,
    )

def train(cfg: TrainConfig):
    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.tb_dir)
    save_config(cfg, os.path.join(cfg.run_dir, "open_config_curriculum.json"))

    eval_episodes = 20
    teacher       = build_teacher(seed=cfg.seed)
    vec           = DummyVecEnv([make_env_fn(cfg, teacher, render_mode=None) for _ in range(cfg.num_envs)])
    vec           = VecMonitor(vec)

    if cfg.vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_teacher = build_teacher(seed=cfg.seed + 999)
    eval_env     = DummyVecEnv([make_env_fn(cfg, eval_teacher, render_mode=None)])
    eval_env     = VecMonitor(eval_env)
    if cfg.vecnormalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    tb_log = os.path.join(cfg.tb_dir, "tb_open_door_sac_curriculum")
    model = SAC(
        policy         = "MlpPolicy",
        env            = vec,
        verbose        = 1,
        tensorboard_log= tb_log,
        seed           = cfg.seed,
        learning_rate  = cfg.learning_rate,
        buffer_size    = cfg.buffer_size,
        batch_size     = cfg.batch_size,
        gamma          = cfg.gamma,
        tau            = cfg.tau,
        train_freq     = cfg.train_freq,
        gradient_steps = cfg.gradient_steps,
        ent_coef       = cfg.ent_coef,
        policy_kwargs  = dict(net_arch=list(cfg.policy_net_arch)),
    )

    callbacks = [
        TeacherUpdateCallback(teacher=teacher),
        CheckpointCallback(
            save_freq   = max(1, cfg.checkpoint_freq // max(1, cfg.num_envs)),
            save_path   = os.path.join(cfg.run_dir, "checkpoints"),
            name_prefix = "open_door_sac_curriculum",
            save_replay_buffer = True,
            save_vecnormalize  = True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path = cfg.run_dir,
            log_path             = os.path.join(cfg.run_dir, "eval"),
            eval_freq            = max(1, cfg.eval_freq // max(1, cfg.num_envs)),
            n_eval_episodes      = eval_episodes,
            deterministic        = True,
            render               = False,
        ),
    ]

    model.learn(total_timesteps=int(cfg.total_steps), callback=callbacks)
    model.save(os.path.join(cfg.run_dir, "final_model_open_curriculum"))

if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)

"""
	Eval num_timesteps = 2950000, episode_reward = 934.92 +/- 7.26
	Episode length: 501.00 +/- 0.00
	Success rate  : 100.00%
	---------------------------------
	| eval/              |          |
	|    mean_ep_length  | 501      |
	|    mean_reward     | 935      |
	|    success_rate    | 1        |
	| time/              |          |
	|    total_timesteps | 2950000  |
	| train/             |          |
	|    actor_loss      | -2.59    |
	|    critic_loss     | 0.000146 |
	|    ent_coef        | 8.21e-05 |
	|    ent_coef_loss   | 2.23     |
	|    learning_rate   | 0.0003   |
	|    n_updates       | 368737   |
	---------------------------------

	---------------------------------

	---------------------------------
	| rollout/           |          |
	|    ep_len_mean     | 501      |
	|    ep_rew_mean     | 924      |
	|    success_rate    | 0.78     |
	| time/              |          |
	|    episodes        | 5984     |
	|    fps             | 114      |
	|    time_elapsed    | 26026    |
	|    total_timesteps | 2984848  |
	| train/             |          |
	|    actor_loss      | -2.65    |
	|    critic_loss     | 0.00015  |
	|    ent_coef        | 7.39e-05 |
	|    ent_coef_loss   | -6.79    |
	|    learning_rate   | 0.0003   |
	|    n_updates       | 373093   |
	---------------------------------

	---------------------------------

	---------------------------------
	| rollout/           |          |
	|    ep_len_mean     | 501      |
	|    ep_rew_mean     | 920      |
	|    success_rate    | 0.79     |
	| time/              |          |
	|    episodes        | 6008     |
	|    fps             | 114      |
	|    time_elapsed    | 26116    |
	|    total_timesteps | 2996872  |
	| train/             |          |
	|    actor_loss      | -2.61    |
	|    critic_loss     | 0.000108 |
	|    ent_coef        | 7.32e-05 |
	|    ent_coef_loss   | -7.4     |
	|    learning_rate   | 0.0003   |
	|    n_updates       | 374596   |
	---------------------------------

	Eval num_timesteps = 3000000, episode_reward = 909.73 +/- 100.56
	Episode length: 501.00 +/- 0.00
	Success rate  : 95.00%
	---------------------------------
	| eval/              |          |
	|    mean_ep_length  | 501      |
	|    mean_reward     | 910      |
	|    success_rate    | 0.95     |
	| time/              |          |
	|    total_timesteps | 3000000  |
	| train/             |          |
	|    actor_loss      | -2.62    |
	|    critic_loss     | 8.53e-05 |
	|    ent_coef        | 7.24e-05 |
	|    ent_coef_loss   | -2.41    |
	|    learning_rate   | 0.0003   |
	|    n_updates       | 374987   |
	---------------------------------
"""