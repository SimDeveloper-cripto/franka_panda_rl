import time

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.train_open_config import TrainConfig
from open_generalized.env_goal_door import GoalDoorEnv
from open_generalized.teacher import StageTeacher, StageSpec

cfg        = TrainConfig()
RENDER_FPS = 30.0

teacher = StageTeacher(
    stages=(
        StageSpec(
            name="test_max",
            goal_frac_min      = 1.0,
            goal_frac_max      = 1.0,
            friction_scale_min = 1.0,
            friction_scale_max = 1.0,
            damping_scale_min  = 1.0,
            damping_scale_max  = 1.0,
        ),
    ),
    promote_threshold=1.0,
)

env = DummyVecEnv([
    lambda: GoalDoorEnv(
        cfg         = cfg,
        teacher     = teacher,
        render_mode = "human",
    )
])

run_dir = "../runs/door_sac"
env = VecNormalize.load(
    f"{run_dir}/checkpoints/door_sac_curriculum_vecnormalize_3000000_steps.pkl",
    env
)

env.training    = False
env.norm_reward = False

model = SAC.load(
    f"{run_dir}/final_model_curriculum",
    env=env,
)

obs = env.reset()
while True:
    action, _               = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(1.0 / RENDER_FPS)
    if done[0]:
        obs = env.reset()