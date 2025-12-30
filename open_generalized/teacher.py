# teacher.py
from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

@dataclass(frozen=True)
class StageSpec:
    name: str

    # Goal range is expressed as FRACTION of effective range [door_min, door_min + cap]
    goal_frac_min: float
    goal_frac_max: float

    # Physics randomization scales (multiplicative)
    friction_scale_min: float
    friction_scale_max: float
    damping_scale_min : float
    damping_scale_max : float

class StageTeacher:
    """ [stage-based curriculum]
        - samples goal_angle + physics scales
        - advances stage when success-rate over a recent window exceeds threshold
    """
    def __init__(self, stages: Tuple[StageSpec, ...], window_episodes: int = 200, promote_threshold: float = 0.85, seed: int = 0):
        if not stages:
            raise ValueError("stages must be non-empty")
        self._stages    = stages
        self._stage_idx = 0

        self._window                     = int(window_episodes)
        self._promote                    = float(promote_threshold)
        self._recent_success: Deque[int] = deque(maxlen=self._window)

        self._rng = np.random.default_rng(seed)

    @property
    def stage_idx(self) -> int:
        return self._stage_idx

    @property
    def stage_name(self) -> str:
        return self._stages[self._stage_idx].name

    def _maybe_promote(self) -> None:
        if self._stage_idx >= (len(self._stages) - 1):
            return
        if len(self._recent_success) < max(50, self._window // 4):
            return

        sr = float(np.mean(self._recent_success))
        if sr >= self._promote:
            self._stage_idx += 1

            # Reset Window to avoid immediate multi-promotions
            self._recent_success.clear()

    def sample_task(self, door_min: float, effective_max: float) -> Dict[str, float]:
        st = self._stages[self._stage_idx]

        f          = self._rng.uniform(st.goal_frac_min, st.goal_frac_max)
        goal_angle = float(door_min + f * (effective_max - door_min))

        friction_scale = float(self._rng.uniform(st.friction_scale_min, st.friction_scale_max))
        damping_scale  = float(self._rng.uniform(st.damping_scale_min,  st.damping_scale_max))

        return {
            "goal_angle"    : goal_angle,
            "friction_scale": friction_scale,
            "damping_scale" : damping_scale,
            "stage_idx"     : float(self._stage_idx),
        }

    def update(self, success: bool) -> None:
        self._recent_success.append(1 if success else 0)
        self._maybe_promote()

    def stats(self) -> Dict[str, float]:
        sr = float(np.mean(self._recent_success)) if self._recent_success else 0.0
        return {"stage_idx": float(self._stage_idx), "success_rate_window": sr}