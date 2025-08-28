from gitcg_double_mini_gym_env import GITCGDoubleMiniGymEnv
from pettingzoo.test import api_test

env = GITCGDoubleMiniGymEnv()
api_test(env, num_cycles=1000, verbose_progress=False)