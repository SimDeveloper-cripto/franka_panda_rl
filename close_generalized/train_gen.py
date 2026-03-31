# close_generalized/train_gen.py
#
# MODIFICHE RISPETTO ALLA VERSIONE PRECEDENTE
# ============================================
# 1. AdaptiveCurriculumCallback ora monitora anche _grasp_phase:
#    il curriculum NON avanza finché il tasso di grasp confermati è < 0.5.
#    Avanzare il curriculum (porta più lontana, ruotata) prima che il grasp
#    sia stabile distrugge il segnale di reward di fase 1.
#
# 2. GraspDiagnosticCallback: logga su tensorboard la percentuale di episodi
#    in cui il grasp viene raggiunto. È il KPI principale da monitorare:
#    se dopo 500K steps questa metrica è < 0.1, il problema non è il reward
#    ma l'observation space (handle_pos punta al posto sbagliato).
#
# 3. train_gen.py ora istanzia TrainConfig con horizon=400 (era 300).
#    Con horizon=300 la sequenza reach(~50 step)+grasp(~30)+push(~100)+close
#    spesso veniva troncata prima della chiusura, dando un segnale rumoroso.

import numpy as np
import os, sys, time, argparse

from dotenv import load_dotenv
load_dotenv()

if os.name == "nt":
    mujoco_path = os.getenv("MUJOCO_PATH")
    if mujoco_path and os.path.exists(mujoco_path):
        os.add_dll_directory(mujoco_path)

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.train_close_config import TrainConfig

from env_gen import GeneralizedDoorEnv
from train_close import SuccessRateCallback


# ─────────────────────────────────────────────────────────────────────────────
class GraspDiagnosticCallback(BaseCallback):
    """
    Monitora e logga su TensorBoard la frequenza con cui il grasp viene
    confermato durante il training. È il KPI fondamentale:

    - grasp_rate < 0.05 dopo 300K steps → handle_pos è probabilmente sbagliato
    - grasp_rate 0.05–0.30 → reward di fase 1 funziona, manca esplorazione
    - grasp_rate > 0.50 → sistema sano, monitorare success_rate
    """
    def __init__(self, log_every: int = 10_000):
        super().__init__()
        self.log_every  = log_every
        self.grasps     = 0
        self.episodes   = 0
        self._was_grasp = {}   # env_idx → bool (era in grasp_phase al passo precedente)

    def _on_step(self) -> bool:
        # Legge lo stato FSM da ogni env parallelo
        try:
            grasp_phases = self.training_env.get_attr("_grasp_phase")
            dones        = self.locals.get("dones", [False] * len(grasp_phases))

            for i, (gp, done) in enumerate(zip(grasp_phases, dones)):
                prev = self._was_grasp.get(i, False)
                # Transizione False→True = nuovo grasp confermato
                if gp and not prev:
                    self.grasps += 1
                if done:
                    self.episodes += 1
                self._was_grasp[i] = gp if not done else False

        except Exception:
            pass  # VecEnv potrebbe non supportare get_attr in tutti i contesti

        if self.n_calls % self.log_every == 0 and self.episodes > 0:
            grasp_rate = self.grasps / max(1, self.episodes)
            self.logger.record("custom/grasp_rate", grasp_rate)
            self.logger.record("custom/grasp_count", self.grasps)
            self.logger.record("custom/episodes", self.episodes)
            self.grasps   = 0
            self.episodes = 0

        return True


# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveCurriculumCallback(BaseCallback):
    """
    Avanza il curriculum solo se ENTRAMBE le condizioni sono soddisfatte:
    1. success_rate > 0.85 (porta chiusa)
    2. grasp_rate > 0.50 (il grasp viene usato nella maggioranza degli episodi)

    Senza la condizione 2, il curriculum avanza anche quando il robot sta
    chiudendo la porta per pushing, e la generalizzazione non avviene.
    """
    def __init__(
        self,
        success_callback: SuccessRateCallback,
        grasp_callback:   GraspDiagnosticCallback,
        check_freq: int = 25_000
    ):
        super().__init__()
        self.success_cb = success_callback
        self.grasp_cb   = grasp_callback
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            sr = self.success_cb.successes / max(1, self.success_cb.episodes)
            gr = self.grasp_cb.grasps     / max(1, self.grasp_cb.episodes)
            current_level = self.training_env.get_attr("curriculum_level")[0]

            # Entrambe le metriche devono essere soddisfatte
            if sr > 0.85 and gr > 0.50 and current_level < 1.0:
                new_level = min(1.0, current_level + 0.05)
                self.training_env.env_method("set_curriculum_level", new_level)
                self.success_cb.successes = 0
                self.success_cb.episodes  = 0
                print(f"\n[CURRICULUM] Level Up → {new_level:.2f}  "
                      f"(success={sr:.2f}, grasp={gr:.2f})")
            elif sr > 0.85 and gr <= 0.50:
                print(f"\n[CURRICULUM] Bloccato: success={sr:.2f} ok ma "
                      f"grasp_rate={gr:.2f} < 0.50. Il robot sta ancora usando pushing.")

        return True


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play",  action="store_true")
    parser.add_argument("--model", type=str, default="runs/close_gen/best_model.zip")
    args = parser.parse_args()

    # horizon=400: vedi commento in cima al file
    my_cfg = TrainConfig(run_dir="runs/close_gen", total_steps=3_000_000, num_envs=4, horizon=400)

    if args.play:
        env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg, render_mode="human")])
        env.env_method("set_curriculum_level", 1.0)

        vn_path = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        if os.path.exists(vn_path):
            env             = VecNormalize.load(vn_path, env)
            env.training    = False
            env.norm_reward = False

        model = SAC.load(args.model, env=env)
        obs   = env.reset()

        prev_action = np.zeros(env.action_space.shape)
        alpha       = 0.5
        target_dt   = 1.0 / my_cfg.control_freq

        print("[INFO] Playing in Real-Time...")
        while True:
            start_t   = time.perf_counter()
            action, _ = model.predict(obs, deterministic=True)
            action    = alpha * action + (1.0 - alpha) * prev_action
            prev_action = action.copy()
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, _, terminated, truncated, _ = step_result
                done = np.logical_or(terminated, truncated)
            else:
                obs, _, done, _ = step_result

            env.render()

            elapsed = time.perf_counter() - start_t
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

            if np.any(done):
                obs = env.reset()
                prev_action[:] = 0

    else:
        os.makedirs(my_cfg.run_dir, exist_ok=True)

        env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg) for _ in range(my_cfg.num_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        scb = SuccessRateCallback(log_every=10_000)
        gcb = GraspDiagnosticCallback(log_every=10_000)
        ccb = AdaptiveCurriculumCallback(success_callback=scb, grasp_callback=gcb)

        from stable_baselines3.common.callbacks import EvalCallback
        from train_close import SaveVecNormalizeCallback

        eval_env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg)])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
        eval_env.obs_rms = env.obs_rms

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=my_cfg.run_dir,
            log_path=os.path.join(my_cfg.run_dir, "eval"),
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVecNormalizeCallback(
                save_path=os.path.join(my_cfg.run_dir, "vecnormalize.pkl")
            ),
        )

        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=my_cfg.tb_dir,
            learning_rate=my_cfg.learning_rate,
            buffer_size=my_cfg.buffer_size,
            batch_size=my_cfg.batch_size,
            gamma=my_cfg.gamma,
            tau=my_cfg.tau,
            train_freq=my_cfg.train_freq,
            gradient_steps=my_cfg.gradient_steps,
            learning_starts=my_cfg.learning_starts,
            ent_coef=my_cfg.ent_coef,
            policy_kwargs=dict(net_arch=list(my_cfg.policy_net_arch)),
        )

        model.learn(
            total_timesteps=my_cfg.total_steps,
            callback=[scb, gcb, ccb, eval_cb]
        )
        model.save(os.path.join(my_cfg.run_dir, "best_model"))
        env.save(os.path.join(my_cfg.run_dir, "vecnormalize.pkl"))


if __name__ == "__main__":
    main()