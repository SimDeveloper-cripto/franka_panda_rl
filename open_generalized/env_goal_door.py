# open_generalized/env_goal_door.py

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional

from config.train_open_config import TrainConfig
from open_generalized.teacher import StageTeacher

class GoalDoorEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, cfg: TrainConfig, teacher: StageTeacher, render_mode: Optional[str] = None, door_open_cap_rad: float = 0.400):
        super().__init__()
        self.cfg         = cfg
        self.teacher     = teacher
        self.render_mode = render_mode

        self._door_open_cap_rad = float(door_open_cap_rad)

        self._prev_action = None
        self._freeze_next = False

        import robosuite as suite
        from robosuite.controllers import load_composite_controller_config

        controller_config = load_composite_controller_config(controller="BASIC")

        self._rs_env = suite.make(
            env_name              = self.cfg.env_name,
            robots                = self.cfg.robot,
            has_renderer          = (render_mode == "human"),
            has_offscreen_renderer= False,
            use_camera_obs        = False,
            use_object_obs        = True,
            reward_shaping        = False,
            horizon               = self.cfg.horizon,
            control_freq          = self.cfg.control_freq,
            controller_configs    = controller_config,
            ignore_done           = True,
        )

        sim = self._rs_env.sim

        hinge_candidates = [n for n in sim.model.joint_names if ("door" in n.lower() and "hinge" in n.lower())]
        if not hinge_candidates:
            hinge_candidates = [n for n in sim.model.joint_names if ("hinge" in n.lower())]
        if not hinge_candidates:
            raise RuntimeError("Could not find a door hinge joint in MuJoCo model joint_names.")

        self._door_hinge_name = hinge_candidates[0]
        self._door_hinge_jid  = sim.model.joint_name2id(self._door_hinge_name)

        jmin, jmax     = sim.model.jnt_range[self._door_hinge_jid]
        self._door_min = float(jmin)
        self._door_max = float(jmax)
        if (not np.isfinite(self._door_min)) or (not np.isfinite(self._door_max)) or (self._door_max <= self._door_min):
            raise RuntimeError(f"Invalid door hinge limits: [{self._door_min}, {self._door_max}]")

        # Apply (door_min + 0.400 rad) to match your environment constraint
        self._effective_max = float(min(self._door_max, self._door_min + self._door_open_cap_rad))
        if self._effective_max <= self._door_min:
            raise RuntimeError("effective_max <= door_min; check hinge limits or cap.")

        self._door_hinge_qpos_adr = int(sim.model.jnt_qposadr[self._door_hinge_jid])

        # Physics randomization handles (MuJoCo dof arrays)
        self._door_hinge_dof_adr = int(sim.model.jnt_dofadr[self._door_hinge_jid])
        self._base_frictionloss  = float(sim.model.dof_frictionloss[self._door_hinge_dof_adr])
        self._base_damping       = float(sim.model.dof_damping[self._door_hinge_dof_adr])

        obs = self._rs_env.reset()
        if not isinstance(obs, dict):
            raise RuntimeError("Expected robosuite observation to be a dict.")

        self._obs_keys         = self._select_obs_keys(obs)
        flat                   = self._flatten_obs(obs, goal_norm=0.0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32)

        low, high         = self._rs_env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        # Episode state
        self._step_count                       = 0
        self._prev_door_angle: Optional[float] = None
        self._success_latched                  = False

        self._start_eef_pos: Optional[np.ndarray] = None
        self._return_hold                         = 0

        # Current sampled task
        self._goal_angle    : float = float(self._door_min)
        self._friction_scale: float = 1.0
        self._damping_scale : float = 1.0

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

    def _flatten_obs(self, obs: Dict[str, Any], goal_norm: float) -> np.ndarray:
        parts = [obs[k].ravel().astype(np.float32) for k in self._obs_keys]
        base  = np.concatenate(parts, axis=0)
        g     = np.array([np.clip(goal_norm, 0.0, 1.0)], dtype=np.float32)
        return np.concatenate([base, g], axis=0)

    def _get_door_angle(self) -> float:
        a = float(self._rs_env.sim.data.qpos[self._door_hinge_qpos_adr])
        return float(np.clip(a, self._door_min, self._effective_max))

    def _apply_physics_randomization(self) -> None:
        sim                                                  = self._rs_env.sim
        sim.model.dof_frictionloss[self._door_hinge_dof_adr] = self._base_frictionloss * float(self._friction_scale)
        sim.model.dof_damping[self._door_hinge_dof_adr]      = self._base_damping      * float(self._damping_scale)

    def _goal_norm(self) -> float:
        return float((self._goal_angle - self._door_min) / (self._effective_max - self._door_min))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        obs = self._rs_env.reset()

        # Sample a new task from the teacher and apply it
        task                 = self.teacher.sample_task(door_min=self._door_min, effective_max=self._effective_max)
        self._goal_angle     = float(task["goal_angle"])
        self._friction_scale = float(task["friction_scale"])
        self._damping_scale  = float(task["damping_scale"])
        self._apply_physics_randomization()

        self._start_eef_pos = None
        if isinstance(obs, dict) and "robot0_eef_pos" in obs:
            self._start_eef_pos = obs["robot0_eef_pos"].astype(np.float32).copy()

        self._step_count      = 0
        self._prev_door_angle = self._get_door_angle()

        self._success_latched = False
        self._return_hold     = 0
        self._prev_action     = None
        self._freeze_next     = False

        info = {
            "obs_keys"          : self._obs_keys,
            "door_min"          : self._door_min,
            "door_max"          : self._door_max,
            "effective_max"     : self._effective_max,
            "goal_angle"        : self._goal_angle,
            "goal_norm"         : self._goal_norm(),
            "friction_scale"    : self._friction_scale,
            "damping_scale"     : self._damping_scale,
            "teacher_stage"     : self.teacher.stage_idx,
            "teacher_stage_name": self.teacher.stage_name,
        }
        return self._flatten_obs(obs, goal_norm=self._goal_norm()), info

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

        obs, _, rs_done, rs_info = self._rs_env.step(action)

        door_angle            = self._get_door_angle()
        prev                  = float(self._prev_door_angle) if (self._prev_door_angle is not None) else door_angle
        delta                 = float(door_angle - prev)
        self._prev_door_angle = door_angle

        denom            = max(1e-6, (self._goal_angle - self._door_min))
        progress_to_goal = float(np.clip((door_angle - self._door_min) / denom, 0.0, 1.0))
        is_success       = bool(door_angle >= self._goal_angle)

        r = 0.0
        r += float(self.cfg.w_progress) * progress_to_goal
        r += float(self.cfg.w_delta) * float(delta)

        w_act = float(self.cfg.w_action_post_success) if self._success_latched else float(self.cfg.w_action)
        r     -= w_act * float(np.linalg.norm(action))
        r     -= float(self.cfg.time_penalty)

        if is_success and not self._success_latched:
            r += float(self.cfg.success_bonus)
            self._success_latched = True

        # --- Post-success shaping: keep door open + return to start ---
        if self.cfg.enable_return_stage and self._success_latched:
            regress = max(0.0, -float(delta))
            r       -= float(self.cfg.w_door_regress) * regress

            if isinstance(obs, dict) and (self._start_eef_pos is not None) and ("robot0_eef_pos" in obs):
                cur  = obs["robot0_eef_pos"].astype(np.float32)
                dist = float(np.linalg.norm(cur - self._start_eef_pos))

                # reward return proximity (smooth)
                if dist < float(self.cfg.return_pos_tol):
                    r += float(self.cfg.w_return_pos) * (1.0 - dist / float(self.cfg.return_pos_tol))
                    self._return_hold += 1
                else:
                    self._return_hold = 0

        self._freeze_next = bool(self.cfg.freeze_on_return and (self._return_hold >= int(self.cfg.freeze_min_hold)))

        reward     = float(np.clip(r, -10.0, 10.0))
        terminated = bool(is_success and self.cfg.terminate_on_success)
        truncated  = bool(self._step_count >= int(self.cfg.horizon))

        if self.cfg.enable_return_stage and self._success_latched:
            if self._return_hold >= int(self.cfg.return_hold_steps):
                terminated = True

        if bool(rs_done) and not terminated:
            truncated = True

        self._step_count += 1

        info = {
            "door_angle"        : float(door_angle),
            "goal_angle"        : float(self._goal_angle),
            "goal_norm"         : float(self._goal_norm()),
            "progress_to_goal"  : float(progress_to_goal),
            "delta"             : float(delta),
            "is_success"        : bool(is_success),
            "success_latched"   : bool(self._success_latched),
            "teacher_stage"     : self.teacher.stage_idx,
            "teacher_stage_name": self.teacher.stage_name,
        }

        if isinstance(rs_info, dict):
            info.update({f"rs_{k}": v for k, v in rs_info.items()})

        return self._flatten_obs(obs, goal_norm=self._goal_norm()), reward, terminated, truncated, info