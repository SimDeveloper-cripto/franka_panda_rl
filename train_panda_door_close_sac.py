#!/usr/bin/env python3
# train_panda_door_close_sac.py
# Franka Panda closes a door in robosuite (Door Task) using SAC (Stable-Baselines3 + Gymnasium)

# TODO LA PORTA SI CHIUDA MA LA CHIUSURA E' TROPPO AGGRESSIVA E NON EFFICACE NEL COME AVVIENE
# TODO UNA VOLTA CHIUSA LA PORTA NON DEVE PIU' FARE MOVIMENTI INUTILI, AL MASSIMO RITORNARE INDIETRO

from __future__ import annotations

import os
import json
import argparse
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback, CallbackList


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except Exception as e:
        print(e)
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
    terminate_on_success: bool  = True  # Quando la porta è chiusa --> episodio finito.

    num_envs    : int  = 8
    vecnormalize: bool = True

    # SAC hyperparams
    total_steps    : int   = 3_000_000
    learning_rate  : float = 3e-4
    buffer_size    : int   = 1_000_000
    batch_size     : int   = 256
    gamma          : float = 0.99
    tau            : float = 0.005
    train_freq     : int   = 1
    gradient_steps : int   = 1
    learning_starts: int   = 20_000
    ent_coef       : str   = "auto"
    policy_net_arch: Tuple[int, int] = (256, 256)

    eval_freq      : int = 50_000
    n_eval_episodes: int = 10
    checkpoint_freq: int = 200_000

    close_fraction: float = 0.08

    init_open_min_fraction: float = 0.70
    init_open_max_fraction: float = 1.00

    w_progress   : float = 2.0
    w_delta      : float = 4.0
    w_action     : float = 0.05
    time_penalty : float = 0.002
    success_bonus: float = 5.0

    debug_print_every: int = 200


class RoboSuiteDoorCloseGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg         = cfg
        self.render_mode = render_mode

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(env_name=cfg.env_name, robots=cfg.robot, controller_configs=controller_config, has_renderer=(render_mode == "human"),
            has_offscreen_renderer=False, use_camera_obs=cfg.use_camera_obs, use_object_obs=cfg.use_object_obs, reward_shaping=cfg.reward_shaping,
            reward_scale=cfg.reward_scale, horizon=cfg.horizon, control_freq=cfg.control_freq)

        self._door_hinge_name = "Door_hinge"

        self._door_hinge_qpos_adr: Optional[int] = None
        for name, adr in zip(self._rs_env.sim.model.joint_names, self._rs_env.sim.model.jnt_qposadr):
            if name == self._door_hinge_name:
                self._door_hinge_qpos_adr = int(adr)
                break
        if self._door_hinge_qpos_adr is None:
            raise RuntimeError("Door_hinge joint not found! Check robosuite model joint names.")

        self._door_hinge_dof_adr: Optional[int] = None
        try:
            jid = self._rs_env.sim.model.joint_name2id(self._door_hinge_name)
            self._door_hinge_dof_adr = int(self._rs_env.sim.model.jnt_dofadr[jid])
        except Exception as e:
            print(str(e))
            self._door_hinge_dof_adr = None

        jid        = self._rs_env.sim.model.joint_name2id(self._door_hinge_name)
        jmin, jmax = self._rs_env.sim.model.jnt_range[jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)
        if not np.isfinite(self._door_min) or not np.isfinite(self._door_max) or self._door_max <= self._door_min:
            raise RuntimeError(f"Invalid door hinge limits: [{self._door_min}, {self._door_max}]")

        rng                   = (self._door_max - self._door_min)
        self._success_angle   = self._door_min + float(cfg.close_fraction) * rng
        self._success_latched = False

        low, high         = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        obs            = self._rs_env.reset()
        self._obs_keys = self._select_obs_keys(obs)
        flat           = self._flatten_obs(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

        self._step_count = 0
        self._prev_door_angle: Optional[float] = None

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
        assert self._door_hinge_qpos_adr is not None
        a = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        return float(np.clip(a, self._door_min, self._door_max))

    def _force_door_angle(self, angle: float) -> None:
        assert self._door_hinge_qpos_adr is not None
        a = float(np.clip(angle, self._door_min, self._door_max))
        self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr] = a
        if self._door_hinge_dof_adr is not None:
            try:
                self._rs_env.sim.data.qvel[self._door_hinge_dof_adr] = 0.0
            except Exception as e:
                print(str(e))
                pass
        try:
            self._rs_env.sim.forward()
        except Exception as e:
            print("[ERROR] Some mujoco wrappers might not expose forward; tolerate")
            print(str(e))
            pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        obs                   = self._rs_env.reset()
        self._step_count      = 0
        self._success_latched = False

        rng = (self._door_max - self._door_min)
        lo  = self._door_min + float(self.cfg.init_open_min_fraction) * rng
        hi  = self._door_min + float(self.cfg.init_open_max_fraction) * rng
        if hi < lo:
            lo, hi = hi, lo
        init_angle = float(np.random.uniform(lo, hi))
        self._force_door_angle(init_angle)

        self._prev_door_angle = self._get_door_angle()

        info = {
            "obs_keys"     : self._obs_keys,
            "door_min"     : self._door_min,
            "door_max"     : self._door_max,
            "success_angle": self._success_angle,
            "init_angle"   : init_angle,
        }
        return self._flatten_obs(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, _rs_reward, rs_done, info = self._rs_env.step(action)
        self._step_count += 1

        door_angle = self._get_door_angle()
        prev_angle = float(self._prev_door_angle) if self._prev_door_angle is not None else door_angle

        delta_close = prev_angle - door_angle
        self._prev_door_angle = door_angle

        if door_angle <= self._success_angle * 1.2:
            delta_close = 0.0

        denom = (self._door_max - self._door_min)

        # progress_close: 0 when fully open, 1 when fully closed
        progress_close = 1.0 - (door_angle - self._door_min) / denom
        progress_close = float(np.clip(progress_close, 0.0, 1.0))

        just_succeeded = False
        if door_angle <= self._success_angle and not self._success_latched:
            self._success_latched = True
            just_succeeded         = True
        # is_success = bool(door_angle <= self._success_angle)
        is_success = self._success_latched

        r = 0.0
        r += self.cfg.w_progress * progress_close
        r += self.cfg.w_delta * float(delta_close)

        r -= self.cfg.w_action * float(np.linalg.norm(action))
        r -= self.cfg.time_penalty

        if just_succeeded:
            r += self.cfg.success_bonus

        reward = float(np.clip(r, -10.0, 10.0))

        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= self.cfg.horizon)

        if bool(rs_done) and not terminated:
            truncated = True

        if self.cfg.debug_print_every > 0 and (self._step_count % self.cfg.debug_print_every == 0):
            print(
                f"[CLOSE] angle={door_angle:.4f} rad "
                f"(min={self._door_min:.4f}, max={self._door_max:.4f}, succ={self._success_angle:.4f}) "
                f"progress={progress_close:.2f} delta={delta_close:+.4f} success={int(is_success)}"
            )

        info = dict(info or {})
        info["is_success"]      = is_success
        info["door_angle"]      = door_angle
        info["door_progress"]   = progress_close
        info["step_count"]      = self._step_count
        info["task"]            = "close"
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

        return True


def make_env_fn(cfg: TrainConfig, render_mode: Optional[str] = None):
    def _init():
        env = RoboSuiteDoorCloseGymnasiumEnv(cfg, render_mode=render_mode)
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
            name_prefix="door_close_sac",
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
        ),
    ]

    model.learn(total_timesteps=cfg.total_steps, callback=CallbackList(callbacks), progress_bar=True)
    model.save(os.path.join(cfg.run_dir, "final_model"))

    if cfg.vecnormalize:
        vec.save(os.path.join(cfg.run_dir, "vecnormalize.pkl"))

    vec.close()
    eval_env.close()


def play(model_path: str, cfg: TrainConfig):
    env  = RoboSuiteDoorCloseGymnasiumEnv(cfg, render_mode="human")
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

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action    = action.astype(np.float32, copy=True)

        action      = alpha * action + (1.0 - alpha) * prev_action
        prev_action = action.copy()

        obs, reward, dones, infos = venv.step(action)
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

    p.add_argument("--play",  action="store_true",  help="Run a trained model with on-screen rendering")
    p.add_argument("--model", type=str, default="", help="Path to model zip (for --play)")

    p.add_argument("--horizon",      type=int, default=500)
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--no-reward-shaping",    action="store_true")
    p.add_argument("--terminate-on-success", action="store_true")

    p.add_argument("--close-fraction",         type=float, default=0.08, help="Success threshold as fraction of hinge range above the minimum (smaller = stricter close)")
    p.add_argument("--init-open-min-fraction", type=float, default=0.70, help="Reset: minimum initial door angle as fraction of hinge range above minimum")
    p.add_argument("--init-open-max-fraction", type=float, default=1.00, help="Reset: maximum initial door angle as fraction of hinge range above minimum")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = TrainConfig(total_steps=int(args.total_steps), num_envs=int(args.num_envs), seed=int(args.seed), run_dir=str(args.run_dir), tb_dir=str(args.tb_dir),
        horizon=int(args.horizon), control_freq=int(args.control_freq), reward_shaping=not args.no_reward_shaping, terminate_on_success=bool(args.terminate_on_success),
        close_fraction=float(args.close_fraction), init_open_min_fraction=float(args.init_open_min_fraction), init_open_max_fraction=float(args.init_open_max_fraction))

    if cfg.init_open_min_fraction < 0.0 or cfg.init_open_max_fraction > 1.0:
        raise SystemExit("init-open fractions must be in [0,1].")
    if cfg.close_fraction < 0.0 or cfg.close_fraction > 1.0:
        raise SystemExit("close-fraction must be in [0,1].")

    if args.play:
        if not args.model:
            raise SystemExit("Missing --model path. Example: --play --model runs/door_close_sac/best_model.zip")
        play(args.model, cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()