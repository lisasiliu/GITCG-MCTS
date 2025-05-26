import supersuit as ss
import gymnasium as gym
import numpy as np
from pettingzoo.utils import wrappers
from stable_baselines3 import PPO
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

import wandb, pickle
from wandb.integration.sb3 import WandbCallback

# Initialize a WandB run
wandb.init(
    project="gitcg_ppo_training",
    sync_tensorboard=True,  # Syncs SB3 TensorBoard logs with WandB
    save_code=True          # Saves the training script in WandB
)

model.learn(total_timesteps=100000, # 100000
            callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb.run.id}",
            verbose=1
        )) 
model.save("ppo_gitcg_double_mini")
model = PPO.load("ppo_gitcg_double_mini")

# with open('ppo_double_mini_100000_learn.pkl', 'wb') as f:
#     pickle.dump(model, f)

def evaluate(env, model, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = [False, False]
        while not done[0] or not done[1]:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            print(f"Reward: {reward}, Done: {done}, Info: {info}, Obs: {obs}")
            # current_agent = env.agent_selection # debug
            # action_mask = env.get_action_mask(current_agent) # debug
            # valid_actions = np.where(np.array(action_mask) == 1)[0].tolist()
            # if (len(valid_actions) > 0): # debug
            #     print("agent", current_agent, "picks action", env.action_name(action), "out of", len(valid_actions), "actions")
            # else:
            #     print("agent", current_agent, "has no valid actions")

evaluate(env, model)

