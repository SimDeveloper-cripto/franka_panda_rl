# close_generalized/env_gen.py

import numpy as np
from scipy.spatial.transform import Rotation as r
from train_close import RoboSuiteDoorCloseGymnasiumEnv


class GeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv):
    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        self.curriculum_level = 0.0

        self.door_body_id = self._rs_env.sim.model.body_name2id("Door_main")
        self.base_pos     = np.array(self._rs_env.sim.model.body_pos[self.door_body_id]).copy()
        self.base_quat    = np.array(self._rs_env.sim.model.body_quat[self.door_body_id]).copy()

    def set_curriculum_level(self, level: float):
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def _calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded):
        """
            1. Penalità d'azione costante per favorire l'efficienza.
            2. Bonus statico di chiusura.
            3. Azzeramento dei movimenti superflui una volta completato il task.
        """

        # Otteniamo il reward base (sparse + dense) dalla classe madre
        reward, terminated, truncated = super()._calculate_reward(action, obs, rs_done, door_angle, prev_angle, just_succeeded)

        door_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
        is_closed = abs(door_qpos) < 0.03

        if action is not None:
            reward -= 0.02 * np.linalg.norm(action)
            if is_closed:
                reward  -= 0.5 * np.linalg.norm(action)
                arm_vel = np.linalg.norm(self._rs_env.sim.data.qvel[self._rs_env.robots[0]._ref_joint_vel_indexes])
                if arm_vel < 0.1:
                    reward += 0.5

        return reward, terminated, truncated

    def reset(self, seed=None, options=None):
        p_var = 0.15 * self.curriculum_level
        r_var = 0.30 * self.curriculum_level

        if self.curriculum_level > 0:
            pos_offset    = np.random.uniform(-p_var, p_var, size=3)
            pos_offset[2] = 0
            yaw           = np.random.uniform(-r_var, r_var)
            q_scipy       = r.from_euler('z', yaw).as_quat()

            self._rs_env.sim.model.body_pos[self.door_body_id] = self.base_pos + pos_offset

            q_base = r.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
            q_new  = r.from_quat(q_scipy) * q_base
            res_q  = q_new.as_quat()
            self._rs_env.sim.model.body_quat[self.door_body_id] = np.array([res_q[3], res_q[0], res_q[1], res_q[2]])

        return super().reset(seed=seed, options=options)