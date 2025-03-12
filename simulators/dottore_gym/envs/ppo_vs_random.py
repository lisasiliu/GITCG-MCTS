'''
Testing model (PPO) vs. random moves.
Running games 100x with each as first (first player advantage).
Counting invalid moves.

Player 0: Random
Player 1: MCTS
'''

# from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv
from gitcg_flat_mini_gym_env import GITCGFlatMiniGymEnv
from stable_baselines3 import PPO
import numpy as np
import random, wandb, pickle

# env = GITCGDoubleMiniGymEnv()
env = GITCGFlatMiniGymEnv()
env.reset()

step_to_action = {} # action progression
action_counts = {} # times action is made
p0_wins = 0 # wins
p1_wins = 0
total_games = 1000 # per side

try:
    model = PPO.load("./models/wtb87s1r/model.zip")
    print("loaded existing ppo model")
except FileNotFoundError:
    print("error!!! no ppo model found")
    quit()

wandb.init(project="ppo-vs-random-test")

for game in range(total_games):
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        # print(agent, termination, truncation, observation)
        if termination or truncation:
            # print("TERMIANTE")
            break
        elif agent == "0": # TODO change this to switch the first player
            # ppo model makes model
            action, _states = model.predict(observation, deterministic=True)
            print("agent", agent, "(ppo) picks action", env.action_name(action))
            wandb.log({f"P0 Move {env.turn}": action}, step=game) # messy graphs incoming
        else:
            # random moves from valid moveset
            action_mask = env.get_action_mask(agent)
            print(action_mask)
            valid_actions = np.where(np.array(action_mask) == 1)[0].tolist()
            print(valid_actions)
            if (len(valid_actions) > 0):
                action = random.choice(valid_actions)
                print("agent", agent, "(random) picks action", env.action_name(action))
            else:
                print("agent", agent, "(random) picks None")
                action = None
            wandb.log({f"P1 Move {env.turn}": action}, step=game) # messy graphs incoming
        # print(env.debug_observe(agent))
        if (type(action) == np.ndarray):
            action = action.item()
        if action not in action_counts:
            action_counts[action] = {"0": 0, "1": 0}
        action_counts[action][agent] += 1
        env.step(action)
    print("info:", env.infos)
    print("rewards:", env.rewards)
    print("turn number:", env.turn)
    if (env.infos['0']['status'] == 'loser'):
        p1_wins += 1
    elif (env.infos['0']['status'] == 'winner'):
        p0_wins += 1
    # wandb stats
    wandb.log({"P0 Wins": p0_wins, "P1 Wins": p1_wins}, step=game) # wins
    wandb.log({"P0 Rewards": env.rewards["0"], "P1 Rewards": env.rewards["1"]}, step=game) # rewards
    wandb.log({"Turns": env.turn}, step=game) # turn
    for action, count in action_counts.items(): # actions per game
        wandb.log({f"P0 {env.action_name(int(action))}": count["0"], f"P1 {env.action_name(int(action))}": count["1"]}, step=game)
    env.reset()
env.close()
wandb.finish()
