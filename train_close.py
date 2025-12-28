#!/usr/bin/env python3
# train_close.py

from __future__ import annotations

import os
import time
import json
import argparse
import numpy as np
from dataclasses import asdict
from typing import Dict, Any, Optional

import gymnasium as gym
from gymnasium import spaces
from config.train_close_config import TrainConfig

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

def _has_tensorboard() -> bool:
    try:
        import tensorboard
        return True
    except Exception as e:
        print(str(e))
        return False

class RoboSuiteDoorCloseGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(
            env_name=cfg.env_name,
            robots=cfg.robot,
            controller_configs=controller_config,
            has_renderer=(render_mode == "human"),
            has_offscreen_renderer=False,
            use_camera_obs=cfg.use_camera_obs,
            use_object_obs=cfg.use_object_obs,
            reward_shaping=cfg.reward_shaping,
            reward_scale=cfg.reward_scale,
            horizon=cfg.horizon,
            control_freq=cfg.control_freq,
        )

        self._door_hinge_name = "Door_hinge"
        self._door_hinge_qpos_adr = None
        for name, adr in zip(self._rs_env.sim.model.joint_names, self._rs_env.sim.model.jnt_qposadr):
            if name == self._door_hinge_name:
                self._door_hinge_qpos_adr = int(adr)
                break

        jid = self._rs_env.sim.model.joint_name2id(self._door_hinge_name)
        self._door_hinge_dof_adr = int(self._rs_env.sim.model.jnt_dofadr[jid])

        jmin, jmax = self._rs_env.sim.model.jnt_range[jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)

        rng = self._door_max - self._door_min
        self._success_angle = self._door_min + cfg.close_fraction * rng
        self._success_latched = False

        low, high = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        obs = self._rs_env.reset()
        self._obs_keys = sorted(k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 1)
        flat = self._flatten_obs(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

        self._prev_door_angle      = None
        self._step_count           = 0
        self._post_success_steps   = 0
        self._post_success_horizon = 15

        self._start_eef_pos      = None
        self._start_gripper_qpos = None
        self._return_hold        = 0

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([obs[k].astype(np.float32).ravel() for k in self._obs_keys])

    def _get_door_angle(self) -> float:
        return float(np.clip(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr], self._door_min, self._door_max))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._rs_env.reset()

        self._start_eef_pos      = None
        self._start_gripper_qpos = None
        if isinstance(obs, dict):
            if "robot0_eef_pos" in obs:
                self._start_eef_pos = obs["robot0_eef_pos"].astype(np.float32).copy()
            if "robot0_gripper_qpos" in obs:
                self._start_gripper_qpos = obs["robot0_gripper_qpos"].astype(np.float32).copy()
        self._return_hold = 0

        rng = self._door_max - self._door_min
        lo = self._door_min + self.cfg.init_open_min_fraction * rng
        hi = self._door_min + self.cfg.init_open_max_fraction * rng
        angle = np.random.uniform(lo, hi)

        self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr] = angle
        self._rs_env.sim.forward()

        self._prev_door_angle    = angle
        self._success_latched    = False
        self._post_success_steps = 0
        self._step_count         = 0

        return self._flatten_obs(obs), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, _, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

        door_angle            = self._get_door_angle()
        prev_angle            = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle
        delta_close           = prev_angle - door_angle
        self._prev_door_angle = door_angle

        if door_angle <= self._success_angle:
            delta_close = 0.0

        denom = (self._door_max - self._door_min)

        # progress: 0 open → 1 closed
        progress = 1.0 - (door_angle - self._door_min) / denom
        progress = float(np.clip(progress, 0.0, 1.0))

        just_succeeded = False
        if door_angle <= self._success_angle and not self._success_latched:
            self._success_latched = True
            just_succeeded        = True

        is_success = self._success_latched

        reward = 0.0
        reward += self.cfg.w_progress * progress
        reward += self.cfg.w_delta    * delta_close
        reward -= self.cfg.w_action   * float(np.linalg.norm(action))
        reward -= self.cfg.time_penalty

        if just_succeeded:
            reward += self.cfg.success_bonus

        returned = False
        if self.cfg.enable_return_stage and self._success_latched:
            # 1) discourage re-opening after success
            delta_open = float(door_angle - prev_angle)
            regress    = max(0.0, delta_open)
            reward     -= self.cfg.w_door_regress * regress

            # 2) reward returning end-effector near start
            if isinstance(obs, dict) and (self._start_eef_pos is not None) and ("robot0_eef_pos" in obs):
                cur      = obs["robot0_eef_pos"].astype(np.float32)
                dist     = float(np.linalg.norm(cur - self._start_eef_pos))
                reward   += self.cfg.w_return_pos * float(1.0 - np.tanh(dist / max(1e-6, self.cfg.return_pos_tol)))
                returned = dist < self.cfg.return_pos_tol

            # 3) mild penalty to bring gripper back near initial configuration
            if isinstance(obs, dict) and (self._start_gripper_qpos is not None) and ("robot0_gripper_qpos" in obs):
                gcur    = obs["robot0_gripper_qpos"].astype(np.float32)
                gdist   = float(np.linalg.norm(gcur - self._start_gripper_qpos))
                reward -= 0.1 * gdist

            if returned:
                self._return_hold += 1
            else:
                self._return_hold = 0

        reward     = float(np.clip(reward, -10.0, 10.0))
        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= self.cfg.horizon)

        if self.cfg.enable_return_stage and self._success_latched:
            if self._return_hold >= self.cfg.return_hold_steps:
                terminated = True

        if bool(rs_done) and not terminated:
            truncated = True

        info = dict(info or {})
        info["is_success"] = is_success
        info["door_angle"] = door_angle
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._rs_env.render()
        return None

    def close(self):
        self._rs_env.close()


class SuccessRateCallback(BaseCallback):
    def __init__(self, log_every: int = 5000):
        super().__init__()
        self.log_every  = log_every
        self.successes  = 0
        self.episodes   = 0
        self.ep_success = None

    def _on_training_start(self):
        self.ep_success = np.zeros(self.training_env.num_envs, dtype=bool)

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if info.get("is_success"):
                self.ep_success[i] = True

        for i, d in enumerate(dones):
            if d:
                self.episodes += 1
                if self.ep_success[i]:
                    self.successes += 1
                self.ep_success[i] = False

        if self.num_timesteps % self.log_every == 0:
            self.logger.record("rollout/success_rate", self.successes / max(1, self.episodes))

        return True


def make_env_fn(cfg):
    def _init():
        return Monitor(RoboSuiteDoorCloseGymnasiumEnv(cfg))
    return _init


def train(cfg: TrainConfig):
    os.makedirs(cfg.run_dir, exist_ok=True)
    os.makedirs(cfg.tb_dir, exist_ok=True)

    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    env = DummyVecEnv([make_env_fn(cfg) for _ in range(cfg.num_envs)])
    env = VecMonitor(env)

    if cfg.vecnormalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        learning_starts=cfg.learning_starts,
        ent_coef=cfg.ent_coef,
        policy_kwargs=dict(net_arch=list(cfg.policy_net_arch)),
        tensorboard_log=cfg.tb_dir if _has_tensorboard() else None,
        verbose=1,
        seed=cfg.seed,
    )

    model.learn(cfg.total_steps, callback=SuccessRateCallback())
    model.save(os.path.join(cfg.run_dir, "best_model"))

    if cfg.vecnormalize:
        env.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))

def play(model_path: str, cfg: TrainConfig):
    def make_play_env():
        return Monitor(RoboSuiteDoorCloseGymnasiumEnv(cfg, render_mode="human"))

    venv = DummyVecEnv([make_play_env])

    vn_path = os.path.join(os.path.dirname(model_path), "vecnormalize.pkl")
    if os.path.exists(vn_path):
        venv             = VecNormalize.load(vn_path, venv)
        venv.training    = False
        venv.norm_reward = False

    model = SAC.load(model_path, env=venv)
    model.policy.set_training_mode(False)

    obs = venv.reset()

    prev_action = np.zeros(venv.action_space.shape, dtype=np.float32)
    alpha = 0.2

    target_dt = 1.0 / float(cfg.control_freq)
    next_t    = time.perf_counter()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action    = action.astype(np.float32, copy=True)

        action      = alpha * action + (1.0 - alpha) * prev_action
        prev_action = action.copy()

        obs, rewards, dones, infos = venv.step(action)
        if isinstance(venv, VecNormalize):
            venv.venv.render()
        else:
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
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--run-dir",     type=str, default="runs/door_close_sac")
    p.add_argument("--tb-dir",      type=str, default="runs/tb")

    p.add_argument("--play",  action="store_true")
    p.add_argument("--model", type=str, default="")

    p.add_argument("--horizon",      type=int, default=500)
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--no-reward-shaping",       action="store_true")
    p.add_argument("--no-terminate-on-success", action="store_true")

    p.add_argument("--close-fraction",         type=float, default=0.08)
    p.add_argument("--init-open-min-fraction", type=float, default=0.70)
    p.add_argument("--init-open-max-fraction", type=float, default=1.00)

    return p.parse_args()

def main():
    args = parse_args()

    cfg = TrainConfig(seed=args.seed, run_dir=args.run_dir, tb_dir=args.tb_dir,
        total_steps=args.total_steps, num_envs=args.num_envs, horizon=args.horizon, control_freq=args.control_freq,
        reward_shaping=not args.no_reward_shaping, terminate_on_success=not args.no_terminate_on_success,
        close_fraction=args.close_fraction,
        init_open_min_fraction=args.init_open_min_fraction, init_open_max_fraction=args.init_open_max_fraction)

    if args.play:
        if not args.model:
            raise SystemExit("Missing --model. Example: --play --model runs/door_close_sac/best_model.zip")
        play(args.model, cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()