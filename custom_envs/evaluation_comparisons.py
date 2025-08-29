'''
Evaluate the agents in both discrete and random environments. 
Evaluate the following pairs:
1. Unmasked PPO vs. Masked random
'''

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

'''
Discrete: Unmasked PPO vs. Masked random 
'''
from env_mini_discrete.mini_eval_utils import eval_model_vs_valid_random, show_one_game
print("--- Discrete: Unmasked PPO vs. Masked random ------------------")
model = PPO.load("models/ppo_mini_discrete")
for side in range(2):
    stats = eval_model_vs_valid_random(model, side=str(side), n_games=1000, log_wandb=True)
    print("Printing win stats as side", str(side))
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
for side in range(2): # show the full move-by-move of one game per side
    print("Starting game where PPO is", str(side))
    show_one_game(model, side=str(side))


'''
Discrete: Masked PPO vs. Masked random 
'''
print("--- Discrete: Masked PPO vs. Masked random ------------------")
model = MaskablePPO.load("models/ppo_mini_discrete_masked")
for side in range(2):
    stats = eval_model_vs_valid_random(model, side=str(side), n_games=1000, log_wandb=True)
    print("Printing win stats as side", str(side))
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
for side in range(2): # show the full move-by-move of one game per side
    print("Starting game where PPO is", str(side))
    show_one_game(model, side=str(side))

'''
Random: Unmasked PPO vs. Masked random 
'''
from env_mini_random.mini_eval_utils import eval_model_vs_valid_random, show_one_game
print("--- Random: Unmasked PPO vs. Masked random --------------------")
model = PPO.load("models/ppo_mini_random")
for side in range(2):
    stats = eval_model_vs_valid_random(model, side=str(side), n_games=1000, log_wandb=True)
    print("Printing win stats as side", str(side))
    for key, value in stats.items():
        print(f"{key}: {value}")
    print()
for side in range(2): # show the full move-by-move of one game per side
    print("Starting game where PPO is", str(side))
    show_one_game(model, side=str(side))