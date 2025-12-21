# !/usr/bin/env python3
# train_panda_door_sac.py
# Franka Panda opens a door in robosuite (Door Opening Task) using SAC (Stable-Baselines3 + Gymnasium)

# [RUN]  python train_panda_door_sac.py --total-steps 2000000 --num-envs 2
# [EVAL] python train_panda_door_sac.py --play --model runs/door_sac/best_model.zip

# TODO 0 RAFFINARE L'APERTURA DELLA PORTA (TROVARE MIGLIORE EFFICACIA). CI DEVONO ESSERE MENNO MOVIMENTI INUTILI (ORA NE FA DI MENO).
    # UNA VOLTA APERTA IL BRACCIO DEVE STACCARSI E TORNARE ALLA SUA POSIZIONE INIZIALE
# TODO 1 CHIUSURA PORTA
# TODO 2 UNICA SEQUENZA
# TODO 3 GENERALIZZAZIONE CON CURRICULUM

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


@dataclass
class TrainConfig:
    seed   : int = 123  # or maybe 42 (prob. better)
    run_dir: str = "runs/door_sac"
    tb_dir : str = "runs/tb"

    env_name            : str   = "Door"  # Task
    robot               : str   = "Panda"
    controller          : str   = "OSC_POSE"
    horizon             : int   = 500
    control_freq        : int   = 20
    reward_shaping      : bool  = True
    reward_scale        : float = 1.0
    use_object_obs      : bool  = True
    use_camera_obs      : bool  = False
    terminate_on_success: bool  = False  # !! robosuite typically runs fixed horizon

    num_envs    : int  = 8
    vecnormalize: bool = True

    # SAC Hyperparams
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
    checkpoint_freq: int = 100_000


class RoboSuiteDoorGymnasiumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfig, render_mode: Optional[str] = None):
        super().__init__()
        self.cfg         = cfg
        self.render_mode = render_mode

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller="BASIC")  # Composite controller
        # Panda:
            # OSC_POSE 'arm'
            # GRIP     'gripper'
        # Since version 1.5.1 OSC_POSE doesn't need to be forced

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

        """
        jid = self._rs_env.sim.model.joint_name2id("Door_hinge")
        print("Door hinge limits:", self._rs_env.sim.model.jnt_range[jid]) --> [0.0, 0.418879]
        """
        self._door_hinge_name     = "Door_hinge" # Robosuite limita l'apertura a 22°-23° (0.41-0.42 rad)
        self._door_hinge_qpos_adr = None

        for name, adr in zip(self._rs_env.sim.model.joint_names, self._rs_env.sim.model.jnt_qposadr):
            if name == self._door_hinge_name:
                self._door_hinge_qpos_adr = int(adr)
                break

        if self._door_hinge_qpos_adr is None:
            raise RuntimeError("Door_hinge joint not found!")

        self._door_success_angle = 1.2  # radians ≈ 69°
        self._door_max_angle     = 1.4  # shaping
        self._success_latched    = False

        # Action space
        low, high         = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        obs                    = self._rs_env.reset()
        self._obs_keys         = self._select_obs_keys(obs)
        flat                   = self._flatten_obs(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

        self._step_count = 0

    @staticmethod
    def _select_obs_keys(obs: Dict[str, Any]) -> List[str]:
        keys = []
        for k, v in obs.items():
            if isinstance(v, np.ndarray) and v.dtype != np.object_:
                if v.ndim == 1:
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._success_latched = False

        obs              = self._rs_env.reset()
        self._step_count = 0
        info             = {"obs_keys": self._obs_keys}
        return self._flatten_obs(obs), info

    # Reward & SuccessRate
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if self._success_latched:
            action[:] = 0.0

        obs, reward, done, info = self._rs_env.step(action)
        self._step_count += 1

        """
        is_success = False
        if hasattr(self._rs_env, "_check_success"):
            try:
                is_success = bool(self._rs_env._check_success())  # robosuite envs generally expose _check_success()
            except Exception as e:
                print(str(e))
                is_success = False
        """
        door_angle = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        if door_angle > 0.35:
            reward -= 0.05 * float(np.linalg.norm(action))

        if door_angle < 0.4:
            reward += 2.0 * (door_angle / 0.4)
        else:
            reward += 2.0
            reward += 6.0 * ((door_angle - 0.4) / (self._door_success_angle - 0.4))

        reward = float(np.clip(reward, -10.0, 10.0))

        if door_angle >= self._door_success_angle:
            self._success_latched = True

        if self._step_count % 100 == 0:
            print(
                f"[DOOR] angle={door_angle:.3f} rad  "
                f"progress={(door_angle / self._door_success_angle):.2f}  "
                f"latched={self._success_latched}"
            )

        is_success = self._success_latched
        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= self.cfg.horizon)

        if done and not terminated:
            truncated = True

        info               = dict(info or {})
        info["is_success"] = is_success
        info["step_count"] = self._step_count
        return self._flatten_obs(obs), float(reward), terminated, truncated, info

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

# Callbacks
class SuccessRateCallback(BaseCallback):
    def __init__(self, log_every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every   = log_every
        self._successes  = 0
        self._episodes   = 0
        self._ep_success = None

    def _on_training_start(self) -> None:
        n_envs           = getattr(self.training_env, "num_envs", 1)
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

def make_env_fn(cfg: TrainConfig, rank: int = 0, render_mode: Optional[str] = None):
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

    vec = DummyVecEnv([make_env_fn(cfg, rank=i, render_mode=None) for i in range(cfg.num_envs)])
    vec = VecMonitor(vec)

    if cfg.vecnormalize:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env_fn(cfg, 0)])
    eval_env = VecMonitor(eval_env)
    if cfg.vecnormalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        # Eval stats synced with Training stats
        eval_env.obs_rms = vec.obs_rms

    policy_kwargs = dict(net_arch=list(cfg.policy_net_arch))

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
        tensorboard_log=cfg.tb_dir,
        seed=cfg.seed,
    )

    callbacks = [SuccessRateCallback(log_every=5000), CheckpointCallback(
        save_freq=cfg.checkpoint_freq // max(1, cfg.num_envs),
        save_path=os.path.join(cfg.run_dir, "checkpoints"),
        name_prefix="door_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
    ), EvalCallback(
        eval_env,
        best_model_save_path=cfg.run_dir,
        log_path=os.path.join(cfg.run_dir, "eval"),
        eval_freq=cfg.eval_freq // max(1, cfg.num_envs),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )]

    cb = CallbackList(callbacks)
    model.learn(total_timesteps=cfg.total_steps, callback=cb, progress_bar=True)

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

    arm_scale = np.array([0.15, 0.15, 0.15, 0.25, 0.25, 0.25], dtype=np.float32)
    alpha     = 0.2  # Smoothing (0.1 – 0.3)
    while True:
        action, _ = model.predict(obs, deterministic=True)

        action = action.copy()

        arm_dofs = min(6, action.shape[1])
        action[:, :arm_dofs] *= arm_scale[:arm_dofs]

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
    p.add_argument("--seed",        type=int, default=123)
    p.add_argument("--run-dir",     type=str, default="runs/door_sac")
    p.add_argument("--tb-dir",      type=str, default="runs/tb")

    p.add_argument("--play",  action="store_true",  help="Run a trained model with on-screen rendering")
    p.add_argument("--model", type=str, default="", help="Path to model zip (for --play)")

    # A few env knobs that matter in practice
    p.add_argument("--horizon",      type=int, default=500)
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--no-reward-shaping",    action="store_true")
    p.add_argument("--terminate-on-success", action="store_true")
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
    )

    if args.play:
        if not args.model:
            raise SystemExit("Missing --model path. Example: --play --model runs/door_sac/best_model.zip")
        play(args.model, cfg)
    else:
        train(cfg)

if __name__ == "__main__":
    main()