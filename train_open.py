#!/usr/bin/env python3
# train_open.py
# Franka Panda opens a door in robosuite (Door Opening Task) using SAC (Stable-Baselines3 + Gymnasium)

"""
SAC ottimizza una distribuzione di azioni con entropia: ent_coef = "auto"
Questo significa:
    - Anche a reward quasi nullo continua a esplorare!
    - Se non viene detto esplicitamente “Ora fermati” lui non lo farà mai.
La policy, una volta “soddisfatto” l’obiettivo principale, spesso converge su una zona di quasi-equilibrio (o una postura “inerte”) perché è sufficientemente stabile rispetto a reward/penalità.

Anche con policy deterministica, piccoli jitter possono emergere per:
    (i)   rumore numerico/normalizzazione
    (ii)  reward non “ancorata” a uno stato di stop
    (iii) dinamiche del controller BASIC che reagiscono a micro-variazioni
"""

from __future__ import annotations

import os
import time
import json
import argparse
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, Optional, List

import gymnasium as gym
from gymnasium import spaces
from config.train_open_config import TrainConfig

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList

def _has_tensorboard() -> bool:
    try:
        import tensorboard
        return True
    except Exception as e:
        print(e)
        return False

class RoboSuiteDoorGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfig, render_mode: Optional[str] = None):
        super().__init__()
        self._prev_action = None
        self.cfg          = cfg
        self.render_mode  = render_mode

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(env_name=cfg.env_name, robots=cfg.robot, controller_configs=controller_config,
            has_renderer=(render_mode == "human"), has_offscreen_renderer=False, use_camera_obs=cfg.use_camera_obs, use_object_obs=cfg.use_object_obs,
            reward_shaping=cfg.reward_shaping, reward_scale=cfg.reward_scale, horizon=cfg.horizon, control_freq=cfg.control_freq)

        self._door_hinge_name     = "Door_hinge"
        self._door_hinge_qpos_adr = None
        for name, adr in zip(self._rs_env.sim.model.joint_names, self._rs_env.sim.model.jnt_qposadr):
            if name == self._door_hinge_name:
                self._door_hinge_qpos_adr = int(adr)
                break
        if self._door_hinge_qpos_adr is None:
            raise RuntimeError("Door_hinge joint not found! Check robosuite model joint names.")

        # Read joint limits
        jid            = self._rs_env.sim.model.joint_name2id(self._door_hinge_name)
        jmin, jmax     = self._rs_env.sim.model.jnt_range[jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)
        if not np.isfinite(self._door_min) or not np.isfinite(self._door_max) or self._door_max <= self._door_min:
            raise RuntimeError(f"Invalid door hinge limits: [{self._door_min}, {self._door_max}]")

        self._success_angle   = self._door_min + float(cfg.success_fraction) * (self._door_max - self._door_min)
        self._success_latched = False

        low, high         = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        obs            = self._rs_env.reset()
        self._obs_keys = self._select_obs_keys(obs)
        flat           = self._flatten_obs(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

        self._step_count      = 0
        self._prev_door_angle = None

        # Return stage
        self._start_eef_pos      = None
        self._start_gripper_qpos = None
        self._return_hold        = 0

    @staticmethod
    def _select_obs_keys(obs: Dict[str, Any]) -> List[str]:
        keys: List[str] = []
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.dtype != np.object_ and v.ndim == 1:
                keys.append(k)
        keys.sort()
        if not keys:
            raise RuntimeError(
                "No 1D numpy observations found. "
                "Check robosuite settings (use_object_obs/use_camera_obs) or adjust key selection."
            )
        return keys

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        parts = [obs[k].ravel().astype(np.float32) for k in self._obs_keys]
        return np.concatenate(parts, axis=0)

    def _get_door_angle(self) -> float:
        a = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        return float(np.clip(a, self._door_min, self._door_max))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        obs                   = self._rs_env.reset()

        # Save "start" references for return-stage shaping
        self._start_eef_pos      = None
        self._start_gripper_qpos = None

        if isinstance(obs, dict):
            if "robot0_eef_pos" in obs:
                self._start_eef_pos = obs["robot0_eef_pos"].astype(np.float32).copy()
            if "robot0_gripper_qpos" in obs:
                self._start_gripper_qpos = obs["robot0_gripper_qpos"].astype(np.float32).copy()

        self._return_hold = 0

        self._step_count      = 0
        self._success_latched = False
        self._prev_door_angle = self._get_door_angle()

        info = {
            "obs_keys"     : self._obs_keys,
            "door_min"     : self._door_min,
            "door_max"     : self._door_max,
            "success_angle": self._success_angle,
        }

        self._prev_action = None
        self._freeze_next = False
        return self._flatten_obs(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self._prev_action is None:
            self._prev_action = np.zeros_like(action, dtype=np.float32)
        a                 = float(self.cfg.action_smooth_alpha)
        action            = a * action + (1.0 - a) * self._prev_action
        self._prev_action = action.copy()

        db                          = float(self.cfg.action_deadband)
        action[np.abs(action) < db] = 0.0

        if getattr(self, "_freeze_next", False):
            action[:] = 0.0

        obs, _rs_reward, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

        door_angle = self._get_door_angle()
        prev_angle = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        delta      = door_angle - prev_angle

        self._prev_door_angle = door_angle

        denom    = (self._door_max - self._door_min)
        progress = (door_angle - self._door_min) / denom
        progress = float(np.clip(progress, 0.0, 1.0))

        is_success = bool(door_angle >= self._success_angle)

        r = 0.0
        r += self.cfg.w_progress * progress
        r += self.cfg.w_delta * float(delta)

        w_act = self.cfg.w_action_post_success if self._success_latched else self.cfg.w_action
        r -= w_act * float(np.linalg.norm(action))
        r -= self.cfg.time_penalty

        if is_success and not self._success_latched:
            r += self.cfg.success_bonus
            self._success_latched = True

        # ----------------------------
        # Post-success shaping: keep door open + return to start
        # ----------------------------
        returned = False
        if self.cfg.enable_return_stage and self._success_latched:
            # 1) discourage door "regression"     (closing back after success)
            # closing regression = negative delta (door_angle decreased)
            regress = max(0.0, -float(delta))
            r -= self.cfg.w_door_regress * regress

            # 2) reward returning end-effector near start pose (if available)
            if isinstance(obs, dict) and (self._start_eef_pos is not None) and ("robot0_eef_pos" in obs):
                cur = obs["robot0_eef_pos"].astype(np.float32)
                dist = float(np.linalg.norm(cur - self._start_eef_pos))

                # smooth positive reward, saturates when close
                r += self.cfg.w_return_pos * float(1.0 - np.tanh(dist / max(1e-6, self.cfg.return_pos_tol)))

                returned = dist < self.cfg.return_pos_tol

            # Optional: also encourage gripper to go back near initial configuration (direction-agnostic)
            if isinstance(obs, dict) and (self._start_gripper_qpos is not None) and ("robot0_gripper_qpos" in obs):
                gcur  = obs["robot0_gripper_qpos"].astype(np.float32)
                gdist = float(np.linalg.norm(gcur - self._start_gripper_qpos))
                r -= 0.1 * gdist  # small penalty so it learns to "release"

            # Termination when returned for N consecutive steps
            if returned:
                self._return_hold += 1
            else:
                self._return_hold = 0

        self._freeze_next = self.cfg.freeze_on_return and (self._return_hold >= self.cfg.freeze_min_hold)

        reward = float(np.clip(r, -10.0, 10.0))

        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= self.cfg.horizon)

        if self.cfg.enable_return_stage and self._success_latched:
            if self._return_hold >= self.cfg.return_hold_steps:
                terminated = True

        if bool(rs_done) and not terminated:
            truncated = True

        if self.cfg.debug_print_every > 0 and (self._step_count % self.cfg.debug_print_every == 0):
            print(
                f"[DOOR] angle={door_angle:.4f} rad "
                f"(min={self._door_min:.4f}, max={self._door_max:.4f}, succ={self._success_angle:.4f}) "
                f"progress={progress:.2f} delta={delta:+.4f} success={int(is_success)}"
            )

        info                  = dict(info or {})
        info["is_success"]    = is_success
        info["door_angle"]    = door_angle
        info["door_progress"] = progress
        info["step_count"]    = self._step_count
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return None
        try:
            return self._rs_env.render()
        except Exception as e:
            print(str(e))
            return None

    def close(self):
        try:
            self._rs_env.close()
        except Exception as e:
            print(str(e))
            pass

class SuccessRateCallback(BaseCallback):
    def __init__(self, log_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every   = log_every
        self._successes  = 0
        self._episodes   = 0
        self._ep_success = None

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._ep_success = np.zeros(n_envs, dtype=bool)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)

        if infos:
            for i, info in enumerate(infos):
                if info.get("is_success", False):
                    self._ep_success[i] = True

        if dones is not None:
            dones = np.asarray(dones, dtype=bool)
            for i, d in enumerate(dones):
                if d:
                    self._episodes += 1
                    if self._ep_success[i]:
                        self._successes += 1
                    self._ep_success[i] = False

        if self.num_timesteps > 0 and (self.num_timesteps % self.log_every == 0):
            sr = self._successes / max(1, self._episodes)
            self.logger.record("rollout/success_rate", float(sr))
            self.logger.record("rollout/episodes_counted_for_sr", float(self._episodes))

            # Reset stats to get "windowed" success rate (not cumulative from start)
            self._successes = 0
            self._episodes  = 0

        return True

class SaveVecNormalizeCallback(BaseCallback):
    def __init__(self, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.model.get_vec_normalize_env() is not None:
            try:
                self.model.get_vec_normalize_env().save(self.save_path)
                if self.verbose > 0:
                    print(f"Saved VecNormalize to {self.save_path}")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Failed to save VecNormalize: {e}")
        return True


def make_env_fn(cfg: TrainConfig, render_mode: Optional[str] = None):
    def _init():
        env = RoboSuiteDoorGymnasiumEnv(cfg, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_config(cfg: TrainConfig, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)


def train(cfg: TrainConfig):
    ensure_dir(cfg.run_dir)
    ensure_dir(cfg.tb_dir)
    save_config(cfg, os.path.join(cfg.run_dir, "config.json"))

    vec = DummyVecEnv([make_env_fn(cfg, render_mode=None) for _ in range(cfg.num_envs)])
    vec = VecMonitor(vec)

    if cfg.vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env_fn(cfg, render_mode=None)])
    eval_env = VecMonitor(eval_env)
    if cfg.vecnormalize:
        eval_env         = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = vec.obs_rms

    policy_kwargs = dict(net_arch=list(cfg.policy_net_arch))

    tb_log = cfg.tb_dir if _has_tensorboard() else None
    if tb_log is None:
        print("[INFO] tensorboard not found: disabling tensorboard_log to avoid SB3 crash. "
              "If you want it: pip install tensorboard")

    model = SAC(
        policy="MlpPolicy",
        env=vec,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tb_log,
        seed=cfg.seed,
    )

    callbacks = [
        SuccessRateCallback(log_every=5000),
        CheckpointCallback(
            save_freq=max(1, cfg.checkpoint_freq // max(1, cfg.num_envs)),
            save_path=os.path.join(cfg.run_dir, "checkpoints"),
            name_prefix="open_det",
            save_replay_buffer=True,
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=cfg.run_dir,
            log_path=os.path.join(cfg.run_dir, "eval"),
            eval_freq=max(1, cfg.eval_freq // max(1, cfg.num_envs)),
            n_eval_episodes=cfg.n_eval_episodes,
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVecNormalizeCallback(save_path=os.path.join(cfg.run_dir, "vecnormalize.pkl")) if cfg.vecnormalize else None,
        ),
    ]

    model.learn(total_timesteps=cfg.total_steps, callback=CallbackList(callbacks), progress_bar=True)
    model.save(os.path.join(cfg.run_dir, "final_model"))

    if cfg.vecnormalize:
        vec.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))

    vec.close()
    eval_env.close()

def play(model_path: str, cfg: TrainConfig):
    env  = RoboSuiteDoorGymnasiumEnv(cfg, render_mode="human")
    env  = Monitor(env)
    venv = DummyVecEnv([lambda: env])

    vn_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(vn_path):
        venv             = VecNormalize.load(vn_path, venv)
        venv.training    = False
        venv.norm_reward = False

    model = SAC.load(model_path, env=venv)
    model.policy.set_training_mode(False)
    obs = venv.reset()

    prev_action = np.zeros(venv.action_space.shape, dtype=np.float32)
    alpha       = 0.2

    target_dt = 1.0 / float(cfg.control_freq)
    next_t    = time.perf_counter()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action    = action.astype(np.float32, copy=True)

        # Smoothing
        action      = alpha * action + (1.0 - alpha) * prev_action
        prev_action = action.copy()

        obs, reward, dones, infos = venv.step(action)
        venv.render()

        next_t    += target_dt
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_t = time.perf_counter()

        if np.any(dones):
            obs = venv.reset()
            prev_action[:] = 0.0

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=3_000_000)
    p.add_argument("--num-envs",    type=int, default=8)
    p.add_argument("--seed",        type=int, default=123)
    p.add_argument("--run-dir",     type=str, default="runs/open_det")
    p.add_argument("--tb-dir",      type=str, default="runs/tb")

    p.add_argument("--play",  action="store_true",  help="Run a trained model with on-screen rendering")
    p.add_argument("--model", type=str, default="", help="Path to model zip (for --play)")

    p.add_argument("--horizon",      type=int, default=500)
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--no-reward-shaping",    action="store_true")
    p.add_argument("--terminate-on-success", action="store_true")

    p.add_argument("--success-fraction", type=float, default=0.92, help="Success threshold as fraction of hinge range")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = TrainConfig(
        total_steps=args.total_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        run_dir=args.run_dir,
        tb_dir=args.tb_dir,
        horizon=args.horizon,
        control_freq=args.control_freq,
        reward_shaping=not args.no_reward_shaping,
        terminate_on_success=args.terminate_on_success,
        success_fraction=float(args.success_fraction),
    )

    if args.play:
        if not args.model:
            raise SystemExit("Missing --model path. Example: --play --model runs/open_det/best_model.zip")
        play(args.model, cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()