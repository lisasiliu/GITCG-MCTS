import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import pickle

'''
Double (Flattened) Mini GITCG Setup:
- All Omni Dice
- 1 Character Only (Kaeya)
- Search Space: ~10^5 (similar to tic-tac-toe)
'''

class GITCGFlatMiniGymEnv(AECEnv):
    # setup
    metadata = {
        "render_modes": ["human"],
        "name": "GITCGFlatMiniGymEnv",
        "is_parallelizable": True
    }
    render_mode = "human"
    actions = {
        "kaeya_normal": { "dmg": 2, "dice_cost": 3, "energy": 1},
        "kaeya_skill": { "dmg": 3, "dice_cost": 3, "energy": 1},
        "kaeya_burst": {"dmg": 3, "dice_cost": 3, "energy_cost": 2},
        "broken_rimes_echo": { "atk_discount": 1, "dice_cost": 2, "card_type": "artifact"},
        "hash_brown": { "hp": 2, "dice_cost": 1, "card_type": "food" },
        "sweet_madame": { "hp": 1, "dice_cost": 0, "card_type": "food" },
        "skyward_blade": { "atk_permanent": 1, "atk_per_turn": 1, "dice_cost": 3, "card_type": "weapon"},
        "end_round_action": { "dice_cost": 0}
    }

    def __init__(self):
        super().__init__()
        
        # AEC variables
        self.possible_agents = ["0", "1"] # fixed
        self.agents = ["0", "1"] # current agents

        self._cumulative_rewards = {"0": 0, "1": 0} # for AEC
        self.agent_selection = "0"
        self.terminations = {"0": False, "1": False} # agentID: bool
        self.truncations = {"0": False, "1": False} # agentID: bool
        self.rewards = {"0": 0, "1": 0} # agentID: float

        self.infos = {"0": {}, "1": {}} # agentID: dict[str, Any]
        self.observation_spaces = {} # agentID: space
        self.action_spaces = {} # agentID: space

        # custom variables
        self.vocab = ["broken_rimes_echo", "hash_brown", "sweet_madame", "skyward_blade", "kaeya_normal", "kaeya_skill", "kaeya_burst", "end_round_action"]
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab, start=1)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        self.turn = 1
        self.dice_per_turn = 4

        # flattened space definitions
        self.obs_size = 14 + 7 + len(self.actions)  # 27 in this case
        # 14: max_hp, hp, max_energy, energy, atk_permanent, atk_per_turn_amt, atk_per_turn_used, atk_discount, actions x3, artifact, weapon, full
        # 7: dice, cards x5, declared_end
        # len(self.actions): action mask
        self.obs_space = {
            agent: spaces.Box(low=0, high=max(11, len(self.actions)), shape=(self.obs_size,), dtype=np.float32)
            for agent in self.agents
        }

        self.act_space = {
            agent: spaces.Discrete(len(self.actions))
            for agent in self.agents
        }

    def action_name(self, action): # for debugging
        return self.id_to_word[action+1]
    def action_validity(self, action, agent): # for debugging
        return self.get_action_mask(agent)[action]

    def debug_observe(self, agent):
        return self.observation_spaces[agent]
    def observe(self, agent):
        return self._flatten_observation(agent)
    
    def _flatten_observation(self, agent):
        kaeya = self.observation_spaces[agent]["observation"]["Kaeya"]
        obs = self.observation_spaces[agent]["observation"]
        action_mask = self.get_action_mask(agent)
        flat_obs = np.concatenate([
            [kaeya["max_hp"], kaeya["hp"], kaeya["max_energy"], kaeya["energy"],
             kaeya["atk_permanent"], kaeya["atk_per_turn_amt"], kaeya["atk_per_turn_used"],
             kaeya["atk_discount"], kaeya["actions"][0], kaeya["actions"][1], kaeya["actions"][2],
             kaeya["artifact"], kaeya["weapon"], kaeya["full"]],
            [obs["dice"], obs["cards"][0], obs["cards"][1], obs["cards"][2], obs["cards"][3], obs["cards"][4], obs["declared_end"]],
            action_mask
        ])
        return flat_obs.astype(np.float32)
        
    def last(self, observe=True):
        agent = self.agent_selection
        if observe == False:
            return None, self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        return self._flatten_observation(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def observation_space(self, agent):
        return self.obs_space[agent]
    
    def action_space(self, agent):
        return self.act_space[agent]

    def reset(self, seed=None, options=None):
        print("RESETTING GAME")
        self.observation_spaces = {
            agent: {
                "observation": {
                    "Kaeya": {
                        "max_hp": 10,
                        "hp": 10,
                        "max_energy": 2,
                        "energy": 0,
                        "atk_permanent": 0, # bonus atk 
                        "atk_per_turn_amt": 0, # bonus atk once per turn
                        "atk_per_turn_used": 0,
                        "atk_discount": 0,
                        "actions": np.array([
                                self.word_to_id["kaeya_normal"],
                                self.word_to_id["kaeya_skill"],
                                self.word_to_id["kaeya_burst"]
                            ], dtype=np.int8),
                        "artifact": 0, # None
                        "weapon": 0, # None
                        "full": 0 # False
                    },
                    "dice": self.dice_per_turn,
                    "cards": np.array([
                        self.word_to_id["broken_rimes_echo"],
                        self.word_to_id["hash_brown"],
                        self.word_to_id["sweet_madame"],
                        self.word_to_id["sweet_madame"],
                        self.word_to_id["skyward_blade"]
                    ], dtype=np.int8),
                    "declared_end": int(False),
                },
                "action_mask": np.array([1] * len(self.actions), dtype=np.int8)
            }
            for agent in self.agents
        }

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.action_spaces = {
            agent: list(range(1, len(self.actions)))
            for agent in self.agents
        }

        self.terminations = {"0": False, "1": False} 
        self.truncations = {"0": False, "1": False} 
        self.rewards = {"0": 0, "1": 0}

        self.infos = {"0": {}, "1": {}}
        self.turn = 1 

    def get_action_mask(self, agent):
        # valid - 1, invalid - 0

        mask = [0] * len(self.actions)
        for action in self.actions.keys():
            value = self.actions[action] 
            if "kaeya" in action and self.observation_spaces[agent]["observation"]["Kaeya"]["hp"] <= 0: 
                continue # current character is dead
            if self.observation_spaces[agent]["observation"]["declared_end"]:
                continue # need to end round after end
            if "card_type" in self.actions[action].keys() and self.word_to_id[action] not in self.observation_spaces[agent]["observation"]["cards"]:
                continue  # card not in hand
            if value["dice_cost"] - self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"] > self.observation_spaces[agent]["observation"]["dice"]:
                continue  # not enough dice
            if action.__contains__("burst") and self.observation_spaces[agent]["observation"]["Kaeya"]["max_energy"] != self.observation_spaces[agent]["observation"]["Kaeya"]["energy"]:
                continue # not enough energy
            if "hp" in self.actions[action].keys() and self.observation_spaces[agent]["observation"]["Kaeya"]["full"]:
                continue # full, can't eat more
            mask[self.word_to_id[action] - 1] = 1 # valid otherwise
        return mask

    
    def step(self, action):

        agent = self.agent_selection
        other_agent = "0" if agent == "1" else "1"


        # handling end round
        if (self.observation_spaces[agent]["observation"]["declared_end"] == 1): # skip turn
            self.agent_selection = self._agent_selector.next()
            return
        
        if action == None:
            print("action is", action, "for agent", agent)
            # print(self.observation_spaces[agent])
            # print(self.observation_spaces[other_agent])
            # print("terminations:", self.terminations)
            print("honestly")
            self.reset()
            return

        total_dmg = 0
        action = self.id_to_word[action+1]

        # check if action is valid?
        if self.get_action_mask(agent)[self.word_to_id[action]-1] == 0:
            self.rewards[agent] -= 10 # invalid
            self._cumulative_rewards[agent] += self.rewards[agent]
            print("agent", agent, "gets -10 reward for invalid action")
            action = "end_round_action"

        # subtract dice
        self.observation_spaces[agent]["observation"]["dice"] -= self.actions[action]["dice_cost"]
        if (self.observation_spaces[agent]["observation"]["dice"] < 0): # shouldn't happen
            print("ERROR!! dice less than zero")
            return -1
        
        # apply action
        if action == "end_round_action":
            self.observation_spaces[agent]["observation"]["declared_end"] = 1
            print("agent", agent, "declares end round")
            if (self.observation_spaces[agent]["observation"]["dice"] >= self.dice_per_turn): # didn't use any dice ?
                self.rewards[agent] -= 100 # big penalty
                print("agent", agent, "gets -100 reward for early end round")
        elif "card_type" not in self.actions[action]: # character atk
            self.rewards[agent] += 10 # reward for attacking!
            print("agent", agent, "gets +10 reward for attacking!")
            atk = self.actions[action]["dmg"]
            atk += self.observation_spaces[agent]["observation"]["Kaeya"]["atk_permanent"] 
            
            bonus = self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_amt"]
            used = self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_used"]
            if used == 0:
                self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_used"] = 1
                atk += bonus
            total_dmg = atk
            # add energy
            if "energy" in self.actions[action]:
                self.observation_spaces[agent]["observation"]["Kaeya"]["energy"] += self.actions[action]["energy"]
                if self.observation_spaces[agent]["observation"]["Kaeya"]["energy"] > self.observation_spaces[agent]["observation"]["Kaeya"]["max_energy"]:
                    self.observation_spaces[agent]["observation"]["Kaeya"]["energy"] = self.observation_spaces[agent]["observation"]["Kaeya"]["max_energy"]
            elif self.observation_spaces[agent]["observation"]["Kaeya"]["energy"] == self.observation_spaces[agent]["observation"]["Kaeya"]["max_energy"]:
                self.observation_spaces[agent]["observation"]["Kaeya"]["energy"] = 0 # use burst
            else:
                pass # can't use burst, shouldn't happen
            # deal dmg to opposite character 
            self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"] -= total_dmg
            if self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"] < 0: # no sub-zero sorry
                self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"] = 0
            # print("ATK!!", self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"] + total_dmg, "->", self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"])
        else:
            self.rewards[agent] += 5 # reward for doing something
            print("before", self.observation_spaces[agent]["observation"]["cards"])
            print("agent", agent, "gets +5 reward for doing something! ---- ", action, self.word_to_id[action])
            print()
            self.observation_spaces[agent]["observation"]["cards"] = np.delete(self.observation_spaces[agent]["observation"]["cards"], np.where(self.observation_spaces[agent]["observation"]["cards"] == self.word_to_id[action])[0][0]) if np.any(self.observation_spaces[agent]["observation"]["cards"] == self.word_to_id[action]) else self.observation_spaces[agent]["observation"]["cards"]
            if (len(self.observation_spaces[agent]["observation"]["cards"]) < 5):
                self.observation_spaces[agent]["observation"]["cards"] = np.append(self.observation_spaces[agent]["observation"]["cards"], 0) # filler for env to retain same size
            if "hp" in self.actions[action]:
                cur_hp = self.observation_spaces[agent]["observation"]["Kaeya"]["hp"]
                cur_hp += self.actions[action]["hp"]
                max_hp = self.observation_spaces[agent]["observation"]["Kaeya"]["max_hp"]
                self.observation_spaces[agent]["observation"]["Kaeya"]["hp"] = min(max_hp, cur_hp)
                self.observation_spaces[agent]["observation"]["Kaeya"]["full"] = 1
            if "atk_permanent" in self.actions[action]:
                self.observation_spaces[agent]["observation"]["Kaeya"]["atk_permanent"] += self.actions[action]["atk_permanent"]
            if "atk_per_turn" in self.actions[action]:
                self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_amt"] = self.actions[action]["atk_per_turn"]
                self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_used"] = 0 # once per turn
            if self.actions[action]["card_type"] == "artifact":
                self.observation_spaces[agent]["observation"]["Kaeya"]["artifact"] = self.word_to_id[action]
            if self.actions[action]["card_type"] == "weapon":
                self.observation_spaces[agent]["observation"]["Kaeya"]["weapon"] = self.word_to_id[action]

        # rewards for good actions
        if action in ["kaeya_normal", "kaeya_skill", "kaeya_burst"]:
            self.rewards[agent] += total_dmg + self.observation_spaces[agent]["observation"]["Kaeya"]["hp"] # add current hp to incentivize rounds that end with higher hp
            self._cumulative_rewards[agent] += self.rewards[agent]

        # kills opponent
        if self.observation_spaces[other_agent]["observation"]["Kaeya"]["hp"] <= 0:
            self.rewards[agent] += 100 #big reward
            self._cumulative_rewards[agent] += self.rewards[agent]
            # directly end game here in mini double
            self.terminations[agent] = True 
            self.terminations[other_agent] = True
            self.truncations[agent] = True # for parallel env
            self.truncations[other_agent] = True # for parallel env
            self.infos[agent] = {"status": 'winner'}
            self.infos[other_agent] = {"status": 'loser'}
            self.agent_selection = self._agent_selector.next()
            return

        # check for new round
        if self.observation_spaces[agent]["observation"]["declared_end"] and self.observation_spaces[other_agent]["observation"]["declared_end"]:
            self.observation_spaces[agent]["observation"]["Kaeya"]["atk_per_turn_used"] = 0
            self.observation_spaces[agent]["observation"]["declared_end"] = 0
            self.observation_spaces[agent]["observation"]["Kaeya"]["full"] = 0
            self.observation_spaces[agent]["observation"]["dice"] = 4
            
            self.observation_spaces[other_agent]["observation"]["declared_end"] = 0
            self.observation_spaces[other_agent]["observation"]["declared_end"] = 0
            self.observation_spaces[other_agent]["observation"]["Kaeya"]["full"] = 0
            self.observation_spaces[other_agent]["observation"]["dice"] = 4

            self.turn += 1
            print()
        
        if (self.turn > 15): # end after 15 rounds
            self.terminations[agent] = True 
            self.terminations[other_agent] = True
            self.truncations[agent] = True # for parallel env
            self.truncations[other_agent] = True # for parallel env
            self.rewards[agent] -= 10 # small penalty
            self.rewards[other_agent] -= 10 # small penalty
            self._cumulative_rewards[agent] += self.rewards[agent]
            self._cumulative_rewards[other_agent] += self.rewards[other_agent]
            self.infos[agent] = {"status": 'tie'}
            self.infos[other_agent] = {"status": 'tie'}
            self.agent_selection = self._agent_selector.next()
            return

        # switch player for the next step
        self.agent_selection = self._agent_selector.next()

    def close(self):
        pass