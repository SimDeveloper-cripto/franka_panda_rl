# close_generalized/train_gen.py

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


class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, success_callback: SuccessRateCallback, check_freq=25000):
        super().__init__()
        self.success_cb = success_callback
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            sr = self.success_cb.successes / max(1, self.success_cb.episodes)
            current_level = self.training_env.get_attr("curriculum_level")[0]

            if sr > 0.85 and current_level < 1.0:
                new_level = min(1.0, current_level + 0.05)
                self.training_env.env_method("set_curriculum_level", new_level)

                self.success_cb.successes = 0
                self.success_cb.episodes  = 0
                print(f"\n[CURRICULUM] Level Up: {new_level:.2f} (SR: {sr:.2f}) - Ottimizzazione braccio in corso...")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--model", type=str, default="runs/close_gen/best_model.zip")
    args = parser.parse_args()

    my_cfg = TrainConfig(run_dir="runs/close_gen", total_steps=500_000, num_envs=4)

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
        # smoothing the action
        alpha       = 0.5 # was 1.0 but with this adjustment we use 50% old and 50% new action
        target_dt   = 1.0 / my_cfg.control_freq

        print("[INFO] Playing in Real-Time...")
        while True:
            start_t = time.perf_counter()

            action, _   = model.predict(obs, deterministic=True)
            action      = alpha * action + (1.0 - alpha) * prev_action
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
        # Train
        os.makedirs(my_cfg.run_dir, exist_ok=True)
        env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg) for _ in range(my_cfg.num_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        scb = SuccessRateCallback(log_every=10000)
        ccb = AdaptiveCurriculumCallback(success_callback=scb)

        from stable_baselines3.common.callbacks import EvalCallback
        from train_close import SaveVecNormalizeCallback

        eval_env = DummyVecEnv([lambda: GeneralizedDoorEnv(my_cfg)])
        eval_env = VecMonitor(eval_env)
        if True:
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
            eval_env.obs_rms = env.obs_rms

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=my_cfg.run_dir,
            log_path=os.path.join(my_cfg.run_dir, "eval"),
            eval_freq=10000,
            n_eval_episodes=20,
            deterministic=True,
            render=False,
            callback_on_new_best=SaveVecNormalizeCallback(save_path=os.path.join(my_cfg.run_dir, "vecnormalize.pkl")),
        )

        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=my_cfg.tb_dir)
        model.learn(total_timesteps=my_cfg.total_steps, callback=[scb, ccb, eval_cb])
        model.save(os.path.join(my_cfg.run_dir, "best_model"))
        env.save(os.path.join(my_cfg.run_dir, "vecnormalize.pkl"))


if __name__ == "__main__":
    main()