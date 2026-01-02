# close_generalized/env_gen.py

import numpy as np
from scipy.spatial.transform import Rotation as r
from train_close import RoboSuiteDoorCloseGymnasiumEnv

class GeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        self.curriculum_level = 0.0

        self.door_body_id = self._rs_env.sim.model.body_name2id("Door_main")
        self.base_pos     = np.array(self._rs_env.sim.model.body_pos [self.door_body_id]).copy()
        self.base_quat    = np.array(self._rs_env.sim.model.body_quat[self.door_body_id]).copy()

    def set_curriculum_level(self, level: float):
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        p_var = 0.12 * self.curriculum_level
        r_var = 0.25 * self.curriculum_level

        if self.curriculum_level > 0:
            pos_offset    = np.random.uniform(-p_var, p_var, size=3)
            pos_offset[2] = 0
            yaw           = np.random.uniform(-r_var, r_var)
            q_scipy       = r.from_euler('z', yaw).as_quat()

            # quat_offset = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

            # Apply to MuJoCo model
            self._rs_env.sim.model.body_pos[self.door_body_id] = self.base_pos + pos_offset

            q_base = r.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
            q_new  = r.from_quat(q_scipy) * q_base
            res_q  = q_new.as_quat()
            self._rs_env.sim.model.body_quat[self.door_body_id] = np.array([res_q[3], res_q[0], res_q[1], res_q[2]])

        return super().reset(seed=seed, options=options)