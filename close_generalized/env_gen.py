# close_generalized/env_gen.py
#
# STORIA DEI FIX (cronologica dai log reali)
# ==========================================
#
# FIX 1 — Reward Z esplicito: il robot era sistematicamente 6-8cm sotto la
#   maniglia. Aggiunto -W_Z*|height_diff| in fase 1. Risolto a 200K steps.
#
# FIX 2 — Penalità azione = 0 in fase 1: -0.3/step × 400 step = -120
#   insegnava al robot a stare fermo invece di cercare la maniglia.
#
# FIX 3 — w_action=0 nel config: la classe madre applicava w_action
#   separatamente, vanificando FIX 2 dall'esterno.
#
# FIX 4 — Reward lineare invece di esponenziale: gradiente costante
#   invece di quasi-zero a distanze > 0.10m.
#
# FIX 5 — Penalità hard per gripper aperto dentro tolerance (-15*|grip|):
#   ent_coef era crollato a 0.00015, il robot non esplorava mai grip>0.
#   La penalità negativa crea gradiente senza dipendere dall'esplorazione.
#
# FIX 6 — _GRIPPER_CLOSE_THRESH 0.4→0.6: log mostravano grip=+0.13
#   confermato come grasp. Soglia alzata per richiedere presa reale.
#
# FIX 7 — Reset contatore solo se gripper aperto (non se fuori tolerance
#   con gripper chiuso): quando la porta ruota, la maniglia si sposta e
#   l'eef esce dalla tolerance pur tenendo la presa. Reset duro precedente
#   causava reward hacking: robot restava a dist≈0.03 con grip=+0.99
#   senza mai confermare il grasp, accumulando +10*grip continuamente.
#
# FIX 8 — Reward progress condizionato al grasp in fase 2:
#   base_reward della madre premia il progresso sulla porta anche senza
#   grasp reale. Aggiunto reward amplificato solo quando gripper chiuso
#   E porta si sta chiudendo: segnale pulito che separa pushing da grasp.
#
# FIX 9 — gamma 0.99→0.95 nel config: con gamma=0.99 il critic propaga
#   reward lontani 100+ step quasi ugualmente a reward immediati.
#   La sequenza reach→grasp→close ha credito temporale lungo e il critic
#   fatica. Con 0.95 apprende la causalità più velocemente.

import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from train_close import RoboSuiteDoorCloseGymnasiumEnv


# ─────────────────────────────────────────────────────────────────────────────
# Parametri FSM
# ─────────────────────────────────────────────────────────────────────────────
_GRASP_CONFIRM_STEPS  = 5      # step consecutivi richiesti per confermare grasp
_GRASP_LOSE_STEPS     = 4      # step fuori lose_tol prima di resettare fase 2
_GRIPPER_CLOSE_THRESH = 0.6    # [FIX 6] soglia più alta: richiede presa reale
_GRIPPER_OPEN_THRESH  = -0.3
_APPROACH_HEIGHT_TOL  = 0.015  # m: tolleranza approccio dal basso

# ─────────────────────────────────────────────────────────────────────────────
# Pesi reward — tutti in un posto per debug e tuning rapido
# ─────────────────────────────────────────────────────────────────────────────
# Fase 1
_W_REACH_3D       = 5.0   # reward lineare su dist_handle 3D
_W_REACH_XY       = 3.0   # componente orizzontale separata
_W_REACH_Z        = 6.0   # [FIX 1] componente verticale — segnale mancante
_W_LATERAL_ORI    = 1.5   # bonus allineamento Z in zona ravvicinata
_W_GRIPPER_OPEN   = 1.5   # reward per aprire il gripper durante approach
_W_GRIPPER_CLOSE  = 10.0  # reward per chiudere il gripper dentro tolerance
_W_PUSH_PENALTY   = 5.0   # penalità pressing dall'alto (eef alto + grip chiuso)
_W_APPROACH_BELOW = 3.0   # penalità approccio dal basso
_W_GRASP_BONUS    = 10.0  # bonus one-shot alla conferma del grasp
# Fase 2
_W_GRASP_LOST     = 6.0   # penalità per grasp perso
_W_PROGRESS_GRASP = 5.0   # [FIX 8] amplifica progress porta solo con grasp attivo
_W_ACTION_PHASE2  = 0.005 # penalità azione leggera in fase 2
# Fase 1: penalità azione = 0 [FIX 2]
_W_ACTION_PHASE1  = 0.0


class GeneralizedDoorEnv(RoboSuiteDoorCloseGymnasiumEnv):

    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        self.curriculum_level = 0.0

        self.door_body_id = self._rs_env.sim.model.body_name2id("Door_main")
        self.base_pos     = np.array(self._rs_env.sim.model.body_pos[self.door_body_id]).copy()
        self.base_quat    = np.array(self._rs_env.sim.model.body_quat[self.door_body_id]).copy()

        self.handle_geom_id = None
        for i, name in enumerate(self._rs_env.sim.model.geom_names):
            if "handle" in name.lower():
                self.handle_geom_id = i
                break

        if self.handle_geom_id is not None:
            self.base_friction = self._rs_env.sim.model.geom_friction[self.handle_geom_id].copy()

        # FSM state
        self._grasp_phase         = False
        self._grasp_confirm_count = 0
        self._grasp_lose_count    = 0
        self._return_hold         = 0
        self._diag_step           = 0
        self._prev_door_angle     = None   # [FIX 8] traccia angolo porta per delta

    def set_curriculum_level(self, level: float):
        self.curriculum_level = np.clip(level, 0.0, 1.0)

    def _calculate_reward(self, action, obs, rs_done, door_angle, prev_angle, just_succeeded):

        base_reward, terminated, truncated = super()._calculate_reward(
            action, obs, rs_done, door_angle, prev_angle, just_succeeded
        )

        eef_pos    = obs.get("robot0_eef_pos", np.zeros(3))
        handle_pos = obs.get("handle_pos", obs.get("door_handle_pos", eef_pos))

        dist_handle = np.linalg.norm(eef_pos - handle_pos)
        dist_xy     = np.linalg.norm(eef_pos[:2] - handle_pos[:2])
        height_diff = eef_pos[2] - handle_pos[2]   # positivo = eef sopra maniglia

        door_qpos = self._rs_env.sim.data.qpos[self._rs_env.handle_qpos_addr]
        is_closed = abs(door_qpos) < 0.03

        grip_tol       = getattr(self, "_current_handle_radius", 0.02) + 0.03
        gripper_action = action[-1] if action is not None else 0.0

        gripper_qpos = obs.get("robot0_gripper_qpos")
        if gripper_qpos is not None:
            # np.sum(np.abs(qpos)) garantisce che giunti con segni opposti non si annullino
            gripper_width = np.sum(np.abs(gripper_qpos))
            handle_diameter = getattr(self, "_current_handle_radius", 0.02) * 2.0
            is_physically_closed = gripper_width <= (handle_diameter + 0.015)
        else:
            is_physically_closed = gripper_action > _GRIPPER_CLOSE_THRESH

        # Diagnostica compatta ogni 200 step
        self._diag_step += 1
        if self._diag_step % 200 == 0:
            phase = "GRASP" if self._grasp_phase else "REACH"
            phys_status = "PHYS_OK" if is_physically_closed else "PHYS_OPEN"
            print(
                f"[{phase}] dist={dist_handle:.3f}  "
                f"dXY={dist_xy:.3f}  dZ={height_diff:+.3f}  "
                f"grip={gripper_action:+.2f} ({phys_status})  "
                f"confirm={self._grasp_confirm_count}/{_GRASP_CONFIRM_STEPS}"
            )

        reward = 0.0

        # ══════════════════════════════════════════════════════════════════════
        # FASE 1 — REACH & GRASP
        # base_reward SOPPRESSO (gate anti-pushing)
        # penalità azione = 0 [FIX 2] (il robot deve esplorare liberamente)
        # ══════════════════════════════════════════════════════════════════════
        if not self._grasp_phase and not self._success_latched:

            # Segnale di distanza in tre componenti separate [FIX 1, 4]
            reward -= _W_REACH_3D * dist_handle
            reward -= _W_REACH_XY * dist_xy
            reward -= _W_REACH_Z  * abs(height_diff)   # componente Z mancante

            # Penalità geometriche di approccio
            if height_diff < -_APPROACH_HEIGHT_TOL:
                reward -= _W_APPROACH_BELOW * abs(height_diff + _APPROACH_HEIGHT_TOL)

            if height_diff > 0.03 and gripper_action > 0.2:
                reward -= _W_PUSH_PENALTY * height_diff * gripper_action

            # Penalità allineamento Z in zona ravvicinata (precedentemente bonus)
            if dist_handle < grip_tol * 3.0:
                reward -= _W_LATERAL_ORI * (1.0 - np.exp(-80.0 * height_diff ** 2))
                
                # Penalità allineamento orientamento (Orientation Alignment)
                eef_quat = obs.get("robot0_eef_quat")
                if eef_quat is not None:
                    delta_pos = handle_pos - eef_pos
                    norm_delta = np.linalg.norm(delta_pos)
                    if norm_delta > 0:
                        dir_to_handle = delta_pos / norm_delta
                        rmat = R_scipy.from_quat(eef_quat).as_matrix()
                        # Z locale dell'end-effector
                        eef_z = rmat[:, 2]
                        alignment = abs(np.dot(eef_z, dir_to_handle))
                        # Penalità che cresce se l'end-effector non è allineato all'asse
                        reward -= 2.0 * (1.0 - alignment) * np.exp(-40.0 * dist_handle)
                        
                        # Controllo roll per rendere la presa umana (orizzontale)
                        eef_x = rmat[:, 0]
                        flat_alignment = abs(eef_x[2]) 
                        reward -= 1.5 * flat_alignment * np.exp(-10.0 * dist_handle)

            # Logica gripper (Zero bonus positivi, solo transizioni smussate di penalità)
            if dist_handle > grip_tol * 2.5:
                # Lontano dalla maniglia: il gripper DEVE essere aperto
                if gripper_action > _GRIPPER_OPEN_THRESH:
                    reward -= 5.0 * (gripper_action - _GRIPPER_OPEN_THRESH)
                self._grasp_confirm_count = 0
            elif dist_handle > grip_tol:
                # Zona intermedia di avvicinamento: nessuna penalità sul gripper 
                # per dargli fluidità di transizione da aperto a chiuso
                self._grasp_confirm_count = 0
            else:
                # Dentro tolerance: DEVE essere chiuso
                if gripper_action < _GRIPPER_CLOSE_THRESH:
                    # Penalità proporzionale a quanto il gripper fallisce a stringere
                    reward -= 15.0 * (_GRIPPER_CLOSE_THRESH - gripper_action)

                # Contatore grasp: reset duro se gripper si apre o non stringe fisicamente
                # Richiediamo chiusura meccanica E un centramento millimetrico (dist < 0.035) sulle dita!
                if gripper_action > _GRIPPER_CLOSE_THRESH and is_physically_closed and dist_handle < 0.035:
                    self._grasp_confirm_count += 1
                else:
                    self._grasp_confirm_count = 0

                # Conferma grasp
                if self._grasp_confirm_count >= _GRASP_CONFIRM_STEPS:
                    self._grasp_phase = True
                    self._grasp_lose_count = 0
                    reward += _W_GRASP_BONUS
                    reward += base_reward   # primo step di fase 2

        # ══════════════════════════════════════════════════════════════════════
        # FASE 2 — PUSH TO CLOSE
        # base_reward ATTIVO + amplificatore progress condizionato [FIX 8]
        # ══════════════════════════════════════════════════════════════════════
        elif self._grasp_phase and not self._success_latched:

            reward += base_reward

            # [FIX 8] Reward progress condizionato al grasp.
            if gripper_action > _GRIPPER_CLOSE_THRESH and prev_angle is not None:
                door_progress = prev_angle - door_angle
                if door_progress > 0:
                    reward += _W_PROGRESS_GRASP * door_progress

            # Penalità azione leggera
            if action is not None:
                reward -= _W_ACTION_PHASE2 * np.linalg.norm(action[:-1])

            # [FIX 10] Condizione duale hard per mantenere fase 2.
            # [FIX 11] Addio spinta col polso. Abbassiamo drasticamente le tolleranze di fase 2.
            # Se l'end-effector è a più di 4-5cm dal centro maniglia, la maniglia non è più tra le dita!
            door_moving = (prev_angle is not None and prev_angle - door_angle > 0.001)
            effective_lose_tol = 0.05 if door_moving else 0.04

            gripper_action_lost = gripper_action < _GRIPPER_CLOSE_THRESH
            gripper_lost = gripper_action_lost or not is_physically_closed
            distance_lost = dist_handle > effective_lose_tol

            if gripper_lost or distance_lost:
                # Penalità proporzionale alla gravità della perdita
                if distance_lost:
                    reward -= _W_GRASP_LOST * (dist_handle - effective_lose_tol)
                if gripper_lost:
                    reward -= 5.0 * abs(min(0.0, gripper_action) - _GRIPPER_CLOSE_THRESH)
                # Reset immediato a fase 1
                self._grasp_phase = False
                self._grasp_confirm_count = 0
                self._grasp_lose_count = 0
            else:
                self._grasp_lose_count = 0

            # Mantieni gripper chiuso durante la spinta
            if gripper_action < 0:
                reward -= 3.0 * abs(gripper_action)

            # Anti-sollevamento maniglia
            if action is not None and action[2] > 0.05:
                reward -= 2.0 * action[2]

        # ══════════════════════════════════════════════════════════════════════
        # FASE 3 — RELEASE & HOLD
        # ══════════════════════════════════════════════════════════════════════
        elif self._success_latched:

            reward += base_reward

            if is_closed:
                # Premio stazionario se sta chiusa
                reward += 1.0 - abs(door_qpos)
                if abs(door_qpos) < 0.02:
                    control_freq = self.cfg.control_freq
                    target_hold_steps = int(control_freq * 1.0) # 1 secondo

                    if not hasattr(self, "_hold_closed_duration"):
                        self._hold_closed_duration = 0

                    if self._hold_closed_duration < target_hold_steps:
                        self._hold_closed_duration += 1
                        
                        if gripper_action > _GRIPPER_CLOSE_THRESH:
                            reward += 1.0
                        else:
                            reward -= 2.0 * abs(gripper_action - _GRIPPER_CLOSE_THRESH)
                        
                        if action is not None:
                            action_norm = np.linalg.norm(action[:-1])
                            if action_norm < 0.05:
                                reward += 1.0
                            else:
                                reward -= 2.0 * action_norm
                    else:
                        if gripper_action < _GRIPPER_OPEN_THRESH:
                            reward += 2.0
                            self._return_hold += 1
                            if self._return_hold > 5:
                                reward += 10.0
                                terminated = True
                        else:
                            self._return_hold = 0
                            reward -= 1.0 * abs(gripper_action + 1.0)
                        
                        if action is not None:
                            reward -= 1.0 * np.linalg.norm(action[:-1])
            else:
                self._hold_closed_duration = 0

        return reward, terminated, truncated

    def reset(self, seed=None, options=None):
        # Reset FSM completo a ogni episodio
        self._grasp_phase         = False
        self._grasp_confirm_count = 0
        self._grasp_lose_count    = 0
        self._return_hold         = 0
        self._hold_closed_duration = 0
        self._prev_door_angle     = None
        # _diag_step NON resettato: mostra progresso globale nel log

        if getattr(self, "handle_geom_id", None) is not None:
            base_radius = 0.02
            base_length = 0.08
            r_scale = np.random.uniform(0.7, 1.4)
            l_scale = np.random.uniform(0.8, 1.2)
            self._current_handle_radius = base_radius * r_scale
            if self._rs_env.sim.model.geom_size[self.handle_geom_id] is not None:
                self._rs_env.sim.model.geom_size[self.handle_geom_id][0] = self._current_handle_radius
                self._rs_env.sim.model.geom_size[self.handle_geom_id][1] = base_length * l_scale
        else:
            self._current_handle_radius = 0.02

        p_var = 0.15 * self.curriculum_level
        r_var = 0.30 * self.curriculum_level

        if self.curriculum_level > 0:
            pos_offset    = np.random.uniform(-p_var, p_var, size=3)
            pos_offset[2] = 0
            yaw           = np.random.uniform(-r_var, r_var)
            q_scipy       = R_scipy.from_euler('z', yaw).as_quat()

            self._rs_env.sim.model.body_pos[self.door_body_id] = self.base_pos + pos_offset

            if hasattr(self.cfg, "human_dist_min") and hasattr(self.cfg, "human_dist_max"):
                dist_sample = np.random.uniform(self.cfg.human_dist_min, self.cfg.human_dist_max)
                self._rs_env.sim.model.body_pos[self.door_body_id][0] = dist_sample + pos_offset[0]

            q_base = R_scipy.from_quat([
                self.base_quat[1], self.base_quat[2],
                self.base_quat[3], self.base_quat[0]
            ])
            q_new = R_scipy.from_quat(q_scipy) * q_base
            res_q = q_new.as_quat()
            self._rs_env.sim.model.body_quat[self.door_body_id] = np.array([
                res_q[3], res_q[0], res_q[1], res_q[2]
            ])

        if self.handle_geom_id is not None and getattr(self.cfg, "limit_handle_friction", False):
            max_f     = getattr(self.cfg, "handle_friction_max", 0.8)
            current_f = self.base_friction[0]
            if current_f > max_f:
                self._rs_env.sim.model.geom_friction[self.handle_geom_id][0] = max_f

        return super().reset(seed=seed, options=options)