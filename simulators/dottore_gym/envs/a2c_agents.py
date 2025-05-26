import supersuit as ss
import gymnasium as gym
import numpy as np
from pettingzoo.utils import wrappers
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo.utils.conversions import aec_to_parallel
# from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv
# from gitcg_flat_mini_gym_env import GITCGFlatMiniGymEnv
from gitcg_random_mini_gym_env import GITCGRandomMiniGymEnv


def make_env():
    # env = GITCGFlatMiniGymEnv()
    env = GITCGRandomMiniGymEnv()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = aec_to_parallel(env)
    # env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
    return env

env = make_env()
model = A2C(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=512,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log="./a2c_gitcg_tensorboard/"
)

import wandb
from wandb.integration.sb3 import WandbCallback

# Initialize a WandB run
wandb.init(
    project="gitcg_a2c_training",
    sync_tensorboard=True,  # Syncs SB3 TensorBoard logs with WandB
    save_code=True          # Saves the training script in WandB
)

model.learn(total_timesteps=100000, # 100000
            callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb.run.id}",
            verbose=1
        )) 
model.save("a2c_gitcg_double_mini")
model = A2C.load("a2c_gitcg_double_mini")

def evaluate(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = [False, False]
        while not done[0] or not done[1]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")

evaluate(env, model)

