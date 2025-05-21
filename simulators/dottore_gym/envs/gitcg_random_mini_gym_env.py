import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import pickle, random

'''
Randomized Mini GITCG Setup:
- All Dice are Randomized
- Mulligan Automatic
- 1 Character Only (Kaeya)
- Search Space: 
'''

class GITCGRandomMiniGymEnv(AECEnv):
    # setup
    metadata = {
        "render_modes": ["human"],
        "name": "GITCGFlatMiniGymEnv",
        "is_parallelizable": True
    }
    render_mode = "human"
    actions = {
        "kaeya_normal": { "dmg": 2, "dice_cost": {8: 3}, "energy": 1}, # 3 any (3 total)
        "kaeya_skill": { "dmg": 3, "dice_cost": {7: 1, 8: 2}, "energy": 1}, # 1 cryo, 2 any (3 total)
        "kaeya_burst": {"dmg": 4, "dice_cost": {7: 2, 8: 1}, "energy_cost": 2}, # 2 cryo, 1 any (3 total)
        "broken_rimes_echo": { "atk_discount": 1, "dice_cost": {8: 2}, "card_type": "artifact"},
        "hash_brown": { "hp": 2, "dice_cost": {8: 1}, "card_type": "food" },
        "sweet_madame": { "hp": 1, "dice_cost": {8: 0}, "card_type": "food" },
        "skyward_blade": { "atk_permanent": 1, "atk_per_turn": 1, "dice_cost": {8: 3}, "card_type": "weapon"},
        "end_round_action": { "dice_cost": {8: 0} }
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
        self.dice_type = ["hydro", "pyro", "anemo", "geo", "dendro", "electro", "cryo", "omni"]
        self.dice_to_id = {word: idx for idx, word in enumerate(self.dice_type, start=1)}
        self.id_to_dice = {idx: word for word, idx in self.dice_to_id.items()}
        
        self.turn = 1
        self.dice_per_turn = 4

        # flattened space definitions
        self.obs_size = 14 + 10 + len(self.actions)  # 32 in this case
        # 14: max_hp, hp, max_energy, energy, atk_permanent, atk_per_turn_amt, atk_per_turn_used, atk_discount, actions x3, artifact, weapon, full
        # 7: dice x4, cards x5, declared_end
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
            [obs["dice"][0], obs["dice"][1], obs["dice"][2], obs["dice"][3], 
             obs["cards"][0], obs["cards"][1], obs["cards"][2], obs["cards"][3], obs["cards"][4], obs["declared_end"]],
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
    
    def generate_dice(self, num_dice):
        dice = random.sample(range(1, 9), num_dice)
        for i in range(len(dice)): # automatic mulligan
            if dice[i] != self.dice_to_id["cryo"] and dice[i] != self.dice_to_id["omni"]:
                dice[i] = random.randint(1, 8)
        return dice
    

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
                    "dice": np.array(self.generate_dice(self.dice_per_turn)),
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
            if "card_type" in action and self.word_to_id[action] not in self.observation_spaces[agent]["observation"]["cards"]:
                continue  # card not in hand
            # not enough dice --
            tmp_used_discount = 0
            tmp_used_dice = 0
            enough_dice = True
            for dice, amt in value["dice_cost"].items(): # e.g. 7,1  8,2
                if self.id_to_dice[dice] != "omni": # not omni dice, exact match 
                    if list(self.observation_spaces[agent]["observation"]["dice"]).count(dice) < amt - self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"] + tmp_used_discount:
                        enough_dice = False
                    if list(self.observation_spaces[agent]["observation"]["dice"]).count(dice) < amt:
                        tmp_used_discount = self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"]
                    tmp_used_dice = list(self.observation_spaces[agent]["observation"]["dice"]).count(dice)
                else: # omni dice, any remaining
                    tmp_remaining_dice = len(self.observation_spaces[agent]["observation"]["dice"]) - list(self.observation_spaces[agent]["observation"]["dice"]).count(0)
                    if (tmp_remaining_dice - tmp_used_dice < amt - self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"] + tmp_used_discount):
                        enough_dice = False
            if enough_dice == False:
                continue
            # ------------------
            if action.__contains__("burst") and self.observation_spaces[agent]["observation"]["Kaeya"]["max_energy"] != self.observation_spaces[agent]["observation"]["Kaeya"]["energy"]:
                continue # not enough energy
            if "hp" in action and self.observation_spaces[agent]["observation"]["Kaeya"]["full"]:
                continue # full, can't eat more
            mask[self.word_to_id[action] - 1] = 1 # valid otherwise
        return mask

    
    def step(self, action):

        agent = self.agent_selection
        other_agent = "0" if agent == "1" else "1"

        action_mask = self.get_action_mask(agent)
        valid_actions = np.where(np.array(action_mask) == 1)[0].tolist()
        if (len(valid_actions) > 0): # debug
            print("agent", agent, "picks action", action, "out of", len(valid_actions), "actions")
        else:
            print("agent", agent, "has no valid actions")

        # handling end round
        if (self.observation_spaces[agent]["observation"]["declared_end"] == 1): # skip turn
            self.agent_selection = self._agent_selector.next()
            return
        
        if action == None:
            print("action is", action, "for agent", agent)
            print(self.observation_spaces[agent])
            print(self.observation_spaces[other_agent])
            print("terminations:", self.terminations)
            print("honestly")
            self.reset()
            return

        total_dmg = 0
        action = self.id_to_word[action+1]

        # check if action is valid?
        if self.get_action_mask(agent)[self.word_to_id[action]-1] == 0:
            self.rewards[agent] -= 100 # invalid
            self._cumulative_rewards[agent] += self.rewards[agent]
            print("agent", agent, "gets -100 reward for invalid action")
            action = "end_round_action"

        # subtract dice
        for dice, amt in self.actions[action]["dice_cost"].items(): # e.g. 7,1  8,2
            if self.id_to_dice[dice] != "omni": # not omni dice, exact match 
                count = 0
                for i in range(len(self.observation_spaces[agent]["observation"]["dice"])):
                    if self.observation_spaces[agent]["observation"]["dice"][i] == dice:
                        self.observation_spaces[agent]["observation"]["dice"][i] = 0
                        count += 1
                        if count >= amt - self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"]:
                            break
            else: # omni dice, any remaining
                count = 0
                for i in range(len(self.observation_spaces[agent]["observation"]["dice"])):
                    if self.observation_spaces[agent]["observation"]["dice"][i] != 0:
                        self.observation_spaces[agent]["observation"]["dice"][i] = 0
                        count += 1
                        if count >= amt:
                            break
            
        # self.observation_spaces[agent]["observation"]["dice"] -= self.actions[action]["dice_cost"]
        if (len(self.observation_spaces[agent]["observation"]["dice"]) < self.dice_per_turn): # shouldn't happen
            print("ERROR!! dice less than zero")
            return -1
        
        # apply action
        if action == "end_round_action":
            self.observation_spaces[agent]["observation"]["declared_end"] = 1
            print("agent", agent, "declares end round")
        elif "card_type" not in self.actions[action]: # character atk
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
            if "atk_discount" in self.actions[action]:
                self.observation_spaces[agent]["observation"]["Kaeya"]["atk_discount"] = self.actions[action]["atk_discount"]
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
            self.observation_spaces[agent]["observation"]["dice"] = np.array(self.generate_dice(self.dice_per_turn))
            
            self.observation_spaces[other_agent]["observation"]["declared_end"] = 0
            self.observation_spaces[other_agent]["observation"]["declared_end"] = 0
            self.observation_spaces[other_agent]["observation"]["Kaeya"]["full"] = 0
            self.observation_spaces[other_agent]["observation"]["dice"] = np.array(self.generate_dice(self.dice_per_turn))

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