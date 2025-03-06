'''
Basic double mini setup test.
Running games with valid actions only.
'''

from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv
import numpy as np
import random

env = GITCGDoubleMiniGymEnv()
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break
    else:
        # random moves from valid moveset
        action_mask = env.get_action_mask(agent)
        valid_actions = np.where(np.array(action_mask) == 1)[0].tolist()
        if (len(valid_actions) > 0):
            action = random.choice(valid_actions)
            print("agent", agent, "picks action", env.action_name(action))
        else:
            action = None

    env.step(action)
print("info:", env.infos)
print("rewards:", env.rewards)
print("turn number:", env.turn)
env.close()