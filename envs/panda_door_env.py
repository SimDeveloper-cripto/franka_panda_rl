# envs/panda_door_env.py

import numpy as np
import gymnasium as gym
import robosuite as suite
from gymnasium import spaces


def _flatten_obs(obs_dict: dict) -> np.ndarray:
    parts = []
    for key, value in obs_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 1:
            parts.append(value.astype(np.float32))
    if not parts:
        raise RuntimeError("Nessuna osservazione flat trovata in obs_dict!")
    return np.concatenate(parts, axis=0)


class PandaDoorEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        horizon: int = 250,
        reward_scale: float | None = None,
        reward_shaping: bool = True,
        control_freq: int = 20,
        seed: int | None = None,
        randomize_env: bool = True,
    ):
        super().__init__()

        self.render_mode   = render_mode
        self.horizon       = horizon
        self.randomize_env = randomize_env

        has_renderer           = render_mode == "human"
        has_offscreen_renderer = render_mode == "rgb_array"

        self._env = suite.make(env_name="Door", robots="Panda", use_camera_obs=False, use_object_obs=True,
            has_renderer=has_renderer, has_offscreen_renderer=has_offscreen_renderer, control_freq=control_freq,
            horizon=horizon, reward_scale=reward_scale, reward_shaping=reward_shaping)

        if seed is not None:
            self._env.reset()
            np.random.seed(seed)


        low, high = self._env.action_spec
        self.action_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32)

        obs = self._env.reset()
        flat_obs = _flatten_obs(obs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32)

        self._current_obs = flat_obs
        self._step_count  = 0

    def _get_obs(self):
        return self._current_obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            np.random.seed(seed)

        obs               = self._env.reset()
        flat_obs          = _flatten_obs(obs)
        self._current_obs = flat_obs
        self._step_count  = 0
        return flat_obs, {}

    def step(self, action):
        self._step_count += 1
        obs, reward, done, info = self._env.step(action)
        flat_obs = _flatten_obs(obs)
        self._current_obs = flat_obs

        terminated = bool(done)
        truncated  = self._step_count >= self.horizon
        return flat_obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._env.render()
        elif self.render_mode == "rgb_array":
            return self._env.sim.render(camera_name="frontview", height=256, width=256)

    def close(self):
        self._env.close()