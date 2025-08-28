'''
Evaluate the agents in both discrete and random environments. 
Evaluate the following pairs:
1. Unmasked PPO vs. Masked random
'''

'''
1. Unmasked PPO vs. Masked random
'''
from train_ppo import make_env
from env_mini_discrete.mini_eval_utils import eval_model_vs_valid_random, show_one_game

from stable_baselines3 import PPO

# evaluate 50 games with SB3 as player 0, and log to wandb
model = PPO.load("models/ppo_mini_discrete")
for side in range(2):
    stats = eval_model_vs_valid_random(model, side=str(side), n_games=100, log_wandb=True)
    print("Printing win stats as side", str(side))
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()

# show the full move-by-move of one game as player 1
for side in range(2):
    print("Starting game where PPO is", str(side))
    show_one_game(model, side=str(side))