import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np

'''
Double Mini GITCG Setup:
- All Omni Dice
- 1 Character Only (Kaeya)
- Search Space: ~10^5 (similar to tic-tac-toe)
'''

class GITCGDoubleMiniGymEnv(AECEnv):
    # setup
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
        # AEC variables
        self.possible_agents = [0, 1] # fixed
        self.agents = [0, 1] # current agents

        self.agent_selection = 0
        self.terminations = {0: False, 1: False} # agentID: bool
        self.truncations = {0: False, 1: False} # agentID: bool
        self.rewards = {0: 0, 1: 0} # agentID: float

        self.infos = {0: {}, 1: {}} # agentID: dict[str, Any]
        self.observation_spaces = {} # agentID: space
        self.action_spaces = {} # agentID: space

        # custom variables
        self.vocab = ["broken_rimes_echo", "hash_brown", "sweet_madame", "skyward_blade", "kaeya_normal", "kaeya_skill", "kaeya_burst", "end_round_action"]
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab, start=1)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        
        self.turn = 1
        self.dice_per_turn = 4

        # space definitions
        self.obs_space = {
            agent: spaces.Dict({
                "Kaeya": spaces.Dict({
                    "max_hp": spaces.Discrete(10),
                    "hp": spaces.Discrete(10),
                    "max_energy": spaces.Discrete(2),
                    "energy": spaces.Discrete(2),
                    "atk_permanent": spaces.Discrete(5),
                    "atk_per_turn": spaces.Sequence(  
                            spaces.Tuple((spaces.Discrete(5), spaces.Discrete(2)))  
                        ),
                    "atk_discount": spaces.Discrete(5),
                    "actions": spaces.MultiDiscrete([len(self.vocab)+1] * 3), # normal / skill / burst
                    "artifact": spaces.Discrete(len(self.vocab)+1),
                    "weapon": spaces.Discrete(len(self.vocab)+1),
                    "full": spaces.Discrete(2)
                }),
                "dice": spaces.Discrete(10),
                "cards": spaces.MultiDiscrete([len(self.vocab)+1] * 5), # 5 cards in hand to start
                "declared_end": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(len(self.actions))
            }) 
            for agent in self.agents
        }
        self.act_space = {
            agent: spaces.Discrete(len(self.actions))
            for agent in self.agents
        }
        
        self.reset()
    
    def last(self, observe=True):
        agent = self.agent_selection
        if observe == False:
            return None, self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        return self.observation_spaces[agent], self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def observation_space(self, agent):
        return self.obs_space[agent]
    
    def action_space(self, agent):
        return self.act_space[agent]

    def reset(self, seed=None, options=None):
        self.observation_spaces = {
            agent: {
                "Kaeya": {
                    "max_hp": 10,
                    "hp": 10,
                    "max_energy": 2,
                    "energy": 0,
                    "atk_permanent": 0, # bonus atk 
                    "atk_per_turn": [], # bonus atk once per turn
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
        return self.observation_spaces, self.infos

    def get_action_mask(self, agent):
        # valid - 1, invalid - 0
        mask = [0] * len(self.actions)
        for action in self.actions.keys():
            value = self.actions[action] 
            if "kaeya" in action and self.observation_spaces[agent]["Kaeya"]["hp"] <= 0: 
                continue # current character is dead
            if self.observation_spaces[agent]["declared_end"] and action != "end_round_action":
                continue # need to end round after end
            if "card_type" in action and self.word_to_id[action] not in self.observation_spaces[agent]["cards"]:
                continue  # card not in hand
            if value["dice_cost"] - self.observation_spaces[agent]["Kaeya"]["atk_discount"] > self.observation_spaces[agent]["dice"]:
                continue  # not enough dice
            if action.__contains__("burst") and self.observation_spaces[agent]["Kaeya"]["max_energy"] != self.observation_spaces[agent]["Kaeya"]["energy"]:
                continue # not enough energy
            if "hp" in action and self.observation_spaces[agent]["Kaeya"]["full"]:
                continue # full, can't eat more
            mask[list(self.actions).index(action)] = 1 # valid otherwise
        return mask

    
    def step(self, action):
        if (self.turn > 15):
            self.done = True
            self.truncated = True
        agent = self.agent_selection
        other_agent = (agent + 1) % 2
        total_dmg = 0
        action = self.id_to_word[action+1]
        print(self.observation_spaces[agent])
        print(action, self.observation_spaces[agent]["dice"], agent, self.observation_spaces[agent]["Kaeya"]["hp"], self.observation_spaces[other_agent]["Kaeya"]["hp"], self.terminations)

        # check if action is valid?
        if self.get_action_mask(agent)[list(self.actions).index(action)] == 0:
            self.rewards[agent] -= 100 # invalid
            print("-100 reward")
            action = "end_round_action"

        # subtract dice
        self.observation_spaces[agent]["dice"] -= self.actions[action]["dice_cost"]
        if (self.observation_spaces[agent]["dice"] < 0): # shouldn't happen
            print("ERROR!! dice less than zero")
            return -1
        
        # apply action
        if action == "end_round_action":
            self.observation_spaces[agent]["declared_end"] = True
        elif "card_type" not in self.actions[action]: # character atk
            atk = self.actions[action]["dmg"]
            atk += self.observation_spaces[agent]["Kaeya"]["atk_permanent"] 
            for i in range(len(self.observation_spaces[agent]["Kaeya"]["atk_per_turn"])):
                bonus = self.observation_spaces[agent]["Kaeya"]["atk_per_turn"][i][0]
                used = self.observation_spaces[agent]["Kaeya"]["atk_per_turn"][i][1]
                # print(bonus, used)
                if used == False:
                    self.observation_spaces[agent]["Kaeya"]["atk_per_turn"][i] = (bonus, True)
                    atk += bonus
            total_dmg = atk
            # add energy
            if "energy" in self.actions[action]:
                self.observation_spaces[agent]["Kaeya"]["energy"] += self.actions[action]["energy"]
                self.observation_spaces[agent]["Kaeya"]["energy"] = max(self.observation_spaces[agent]["Kaeya"]["energy"], self.observation_spaces[agent]["Kaeya"]["max_energy"])
            elif self.observation_spaces[agent]["Kaeya"]["energy"] == self.observation_spaces[agent]["Kaeya"]["max_energy"]:
                self.observation_spaces[agent]["Kaeya"]["energy"] = 0 # use burst
            else:
                pass # can't use burst, shouldn't happen
            # deal dmg to opposite character 
            self.observation_spaces[other_agent]["Kaeya"]["hp"] -= total_dmg
            print("ATK!!", self.observation_spaces[other_agent]["Kaeya"]["hp"] + total_dmg, "->", self.observation_spaces[other_agent]["Kaeya"]["hp"])
        else:
            print("before", self.observation_spaces[agent]["cards"])
            self.observation_spaces[agent]["cards"] = np.delete(self.observation_spaces[agent]["cards"], np.where(self.observation_spaces[agent]["cards"] == self.word_to_id[action])[0][0]) if np.any(self.observation_spaces[agent]["cards"] == self.word_to_id[action]) else self.observation_spaces[agent]["cards"]
            if (len(self.observation_spaces[agent]["cards"]) < 5):
                self.observation_spaces[agent]["cards"] = np.append(self.observation_spaces[agent]["cards"], 0) # filler for env to retain same size
            print("after", self.observation_spaces[agent]["cards"])
            if "hp" in self.actions[action]:
                cur_hp = self.observation_spaces[agent]["Kaeya"]["hp"]
                cur_hp += self.actions[action]["hp"]
                max_hp = self.observation_spaces[agent]["Kaeya"]["max_hp"]
                self.observation_spaces[agent]["Kaeya"]["hp"] = min(max_hp, cur_hp)
                self.observation_spaces[agent]["Kaeya"]["full"] = True
            if "atk_permanent" in self.actions[action]:
                self.observation_spaces[agent]["Kaeya"]["atk_permanent"] += self.actions[action]["atk_permanent"]
            if "atk_per_turn" in self.actions[action]:
                self.observation_spaces[agent]["Kaeya"]["atk_per_turn"].append((self.actions[action]["atk_per_turn"], False)) # once per turn
            if self.actions[action]["card_type"] == "artifact":
                self.observation_spaces[agent]["Kaeya"]["artifact"] = action
            if self.actions[action]["card_type"] == "weapon":
                self.observation_spaces[agent]["Kaeya"]["weapon"] = action

        # rewards for good actions
        if action in ["kaeya_normal", "kaeya_skill", "kaeya_burst"]:
            self.rewards[agent] += total_dmg + self.observation_spaces[agent]["Kaeya"]["hp"] # add current hp to incentivize rounds that end with higher hp
        
        # kills opponent
        if self.observation_spaces[other_agent]["Kaeya"]["hp"] <= 0:
            self.rewards[agent] += 100 #big reward
            # directly end game here in mini double
            self.terminations[agent] = True 
            self.terminations[other_agent] = True
            self.infos[agent] = {"status": 'winner'}
            self.infos[other_agent] = {"status": 'loser'}

        # check for new round
        if self.observation_spaces[agent]["declared_end"] and self.observation_spaces[other_agent]["declared_end"]:
            for i in range(len(self.observation_spaces[agent]["Kaeya"]["atk_per_turn"])):
                bonus = self.observation_spaces[agent]["Kaeya"]["atk_per_turn"][i][0]
                self.observation_spaces[agent]["Kaeya"]["atk_per_turn"][i] = (bonus, False)
            self.observation_spaces[agent]["declared_end"] = False
            self.observation_spaces[agent]["full"] = False
            self.observation_spaces[agent]["dice"] = 4
            
            self.observation_spaces[other_agent]["declared_end"] = False
            self.observation_spaces[other_agent]["declared_end"] = False
            self.observation_spaces[other_agent]["full"] = False
            self.observation_spaces[other_agent]["dice"] = 4

            self.turn += 1

        # switch player for the next step
        self.agent_selection = self._agent_selector.next()
        
        return self.observation_spaces, self.rewards, self.terminations, self.truncations, self.infos


    def close(self):
        pass