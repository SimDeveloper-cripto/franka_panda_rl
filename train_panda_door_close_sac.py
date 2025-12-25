#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
from gymnasium import spaces

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


@dataclass
class TrainConfig:
    seed   : int = 42
    run_dir: str = "runs/door_close_sac"
    tb_dir : str = "runs/tb"

    env_name    : str = "Door"
    robot       : str = "Panda"
    horizon     : int = 500
    control_freq: int = 20

    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = True

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
    w_action     : float = 0.05
    time_penalty : float = 0.002
    success_bonus: float = 5.0

    debug_print_every: int = 200


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

    def _flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.concatenate([obs[k].astype(np.float32).ravel() for k in self._obs_keys])

    def _get_door_angle(self) -> float:
        return float(np.clip(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr], self._door_min, self._door_max))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._rs_env.reset()

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

        reward     = float(np.clip(reward, -10.0, 10.0))
        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= self.cfg.horizon)

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
        reward_shaping=not args.no_reward_shaping, terminate_on_success=args.no_terminate_on_success,
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