import supersuit as ss
import gymnasium as gym
import numpy as np
from pettingzoo.utils import wrappers
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from pettingzoo.utils.conversions import aec_to_parallel
# from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv
from gitcg_flat_mini_gym_env import GITCGFlatMiniGymEnv


def make_env():
    env = GITCGFlatMiniGymEnv()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    env = aec_to_parallel(env)
    # env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
    return env

env = make_env()
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=512,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./ppo_gitcg_tensorboard/"
)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_gitcg_double_mini")

# Load the model for inference
model = PPO.load("ppo_gitcg_double_mini")

def evaluate(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}, Info: {info}")

evaluate(env, model)
