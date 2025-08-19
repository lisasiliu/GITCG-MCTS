import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dgisim import LinearEnv, PlayerAction, ActionGenerator 
from typing import Optional

class DottoreLinearGymEnv(gym.Env):
    def __init__(self):
        self.env = LinearEnv()
        self.render_mode = "human"
        self.game_state = self.env.view()[0]

        #   --- TODO can call action_space.sample() to randomly select an action. 
        self.act_gen = self.game_state.action_generator(self.game_state.waiting_for())
        self.action_space = spaces.Discrete(len(self.act_gen.choices()))

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(len(self.env.view()[1]),), dtype=np.float32
        )

    def _get_obs(self):
        """Returns the encoded state as an observation"""
        return np.array(self.env.view()[1], dtype=np.float32)
    
    def _get_info(self):
        """Returns additional info (empty for now)"""
        return {}

    def step(self, action):
        print(self.act_gen.choices()[action])
        print(type(self.act_gen.choices()[action]))
        self.game_state, encoded_state, reward, turn, done = self.env.step(self.act_gen.choices()[action])
    
        observation = self._get_obs()
        info = self._get_info()

        self.act_gen = self.act_gen.choose(self.act_gen.choices()[action])
        self.action_space = spaces.Discrete(len(self.act_gen.choices()))

        return observation, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset()
        return self._get_obs(), self._get_info()
    
    def close():
        pass

    def render(self):
        """Optional render function (modify if visual output is available)"""
        if self.render_mode == "human":
            print("Game State:", self.env.view()[0])