import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import numpy as np
import random

'''
Mini (Flattened) GITCG Setup (random):
- All Dice are Randomized
- Mulligan Automatic (Reroll non-Omni/Cryo dice)
- 1 Character Only (Kaeya)
- Search Space: 
'''

class GITCGRandomMiniGymEnv(AECEnv):
    # setup
    metadata = {
        "render_modes": ["human"],
        "name": "GITCGRandomMiniGymEnv",
        "is_parallelizable": True,
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
        self.possible_agents = ["0", "1"]                   # fixed
        self.agents = ["0", "1"]                            # current agents
        self._cumulative_rewards = {"0": 0.0, "1": 0.0}
        self.agent_selection = "0"
        self.terminations = {"0": False, "1": False}       
        self.truncations = {"0": False, "1": False}
        self.rewards = {"0": 0.0, "1": 0.0}
        self.infos = {"0": {}, "1": {}}                     # agentID: dict[str, Any]

        # custom variables
        self.vocab = ["broken_rimes_echo", "hash_brown", "sweet_madame", "skyward_blade", "kaeya_normal", "kaeya_skill", "kaeya_burst", "end_round_action"]
        self.word_to_id = {word: idx for idx, word in enumerate(self.vocab, start=1)}
        self.id_to_word = {idx: word for word, idx in self.word_to_id.items()}
        self.dice_type = ["hydro", "pyro", "anemo", "geo", "dendro", "electro", "cryo", "omni"]
        self.dice_to_id = {word: idx for idx, word in enumerate(self.dice_type, start=1)}
        self.id_to_dice = {idx: word for word, idx in self.dice_to_id.items()}
        
        self.turn = 1
        self.dice_per_turn = 4

        # ---- observation space definition -------------------------------------------------
        # 14: [max_hp, hp, max_energy, energy, atk_perm, atk_per_turn_amt, atk_per_turn_used,
        #      atk_discount, actions[3], artifact, weapon, full]
        # 10:  [dice[4], cards[5], declared_end]
        # + action mask (len(actions)=8)
        self.obs_size = 14 + 10 + len(self.actions)  # 32
        self.observation_spaces = {                         # agentID: space
            agent: spaces.Box(
                low=0,
                high=max(11, len(self.actions)),
                shape=(self.obs_size,),
                dtype=np.float32,
            )
            for agent in self.agents
        }
        # ---- action space definition ------------------------------------------------------
        self.action_spaces = {                              # agentID: space
            agent: spaces.Discrete(len(self.actions)) for agent in self.agents
        } 
        # internal per-agent state (separate from PettingZoo *.observation_spaces)
        self.state = {}

    # --------- helper functions for debugging ----------------------------------------------
    def action_name(self, action): 
        return self.id_to_word[action+1]
    def action_validity(self, action, agent): 
        return self.get_action_mask(agent)[action]
    def debug_observe(self, agent):
        return self.observation_spaces[agent]
    
    def _effective_dice_cost(self, action_word: str, agent: str) -> dict:
        """
        Return the actual dice cost for an action, applying Kaeya's atk_discount.
        If dice[i] == 0, then that dice has already been used.
        """
        cost = self.actions[action_word]["dice_cost"].copy()
        discount = self.state[agent]["observation"]["Kaeya"]["atk_discount"]
        # apply atk_discount only to character attacks
        if action_word.startswith("kaeya_") and discount > 0:
            if 7 in cost and cost[7] > 0:        # prefer discounting cryo
                cost[7] -= 1
            else:                                # otherwise discount any requirement
                for k in cost:
                    if cost[k] > 0:
                        cost[k] -= 1
                        break
        return {k: v for k, v in cost.items() if v > 0}
         
    # --------- helper functions for gym env -------------------------------------------------

    def _flatten_observation(self, agent):
        kaeya = self.state[agent]["observation"]["Kaeya"]
        obs = self.state[agent]["observation"]
        action_mask = np.array(self.get_action_mask(agent), dtype=np.float32)
        flat_obs = np.concatenate([
            np.array([kaeya["max_hp"], kaeya["hp"], kaeya["max_energy"], kaeya["energy"],
                kaeya["atk_permanent"], kaeya["atk_per_turn_amt"], kaeya["atk_per_turn_used"],
                kaeya["atk_discount"], kaeya["actions"][0], kaeya["actions"][1], kaeya["actions"][2],
                kaeya["artifact"], kaeya["weapon"], kaeya["full"]], dtype=np.float32),
            np.array([obs["dice"][0], obs["dice"][1], obs["dice"][2], obs["dice"][3], 
                obs["cards"][0], obs["cards"][1], obs["cards"][2], obs["cards"][3], obs["cards"][4], 
                obs["declared_end"]], dtype=np.float32),
            action_mask
        ])
        return flat_obs.astype(np.float32)
        
    def last(self, observe=True):
        agent = self.agent_selection
        if observe == False:
            return None, self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]
        return self._flatten_observation(agent), self.rewards[agent], self.terminations[agent], self.truncations[agent], self.infos[agent]

    def observe(self, agent):
        return self._flatten_observation(agent)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    def action_space(self, agent):
        return self.action_spaces[agent]

    # for masked ppo
    def action_masks(self):
        # return valid action mask for the current agent as a boolean array
        agent = self.agent_selection
        return self.valid_action_mask(agent).astype(bool)
    
    def generate_dice(self, num_dice):
        dice = random.sample(range(1, 9), num_dice)
        for i in range(len(dice)): # automatic mulligan
            if dice[i] != self.dice_to_id["cryo"] and dice[i] != self.dice_to_id["omni"]:
                dice[i] = random.randint(1, 8)
        return dice
    

    def reset(self, seed=None, options=None):
        # reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # initial state
        self.state = {
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
                    }, # automatically generate best dice combination for user
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

        self._cumulative_rewards = {"0": 0.0, "1": 0.0}
        self.terminations = {"0": False, "1": False}
        self.truncations = {"0": False, "1": False}
        self.rewards = {"0": 0.0, "1": 0.0}
        self.infos = {"0": {}, "1": {}}
        self.turn = 1

    def get_action_mask(self, agent):
        # 1 = valid, 0 = invalid
        mask = [0] * len(self.actions)
        obs = self.state[agent]["observation"]
        kaeya = obs["Kaeya"]
        for action_word, value in self.actions.items():
            if obs["declared_end"]:
                continue # need to end round after end (env requires an action every turn)
            if action_word.startswith("kaeya_") and kaeya["hp"] <= 0:
                continue # current character is dead
            if "card_type" in value and self.word_to_id[action_word] not in obs["cards"]:
                continue # card not in hand
            eff_cost = self._effective_dice_cost(action_word, agent)
            if not all(sum((d in (dtype,8)) for d in obs["dice"]) >= amt for dtype, amt in eff_cost.items()):
                continue # not enough dice (apply discount only to character attacks)
            if "burst" in action_word and kaeya["energy"] != kaeya["max_energy"]:
                continue # not enough energy for burst
            if "hp" in value and kaeya["full"]:
                continue # full, can't eat more
            mask[self.word_to_id[action_word] - 1] = 1 # valid otherwise
        return mask
    
    def step(self, action):
        agent = self.agent_selection
        other_agent = "0" if agent == "1" else "1"
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(None)
            return
        # if player has already declared end, skip their turn
        if self.state[agent]["observation"]["declared_end"] == 1:
            self.agent_selection = self._agent_selector.next()
            return
        # treat None as "pass/end" (no-op action provided by some libraries)
        if action is None:
            action_word = "end_round_action"
        else:
            action_word = self.id_to_word[action + 1]
        # validate against mask
        if self.get_action_mask(agent)[self.word_to_id[action_word] - 1] == 0:
            self.rewards[agent] -= 10  # invalid -> penalize and convert to end turn
            action_word = "end_round_action"

        total_dmg = 0

        # subtract dice
        eff_cost = self._effective_dice_cost(action_word, agent)
        for dtype, amt in eff_cost.items():
            if dtype != 8:  # exact type (with omni substitution allowed)
                need = amt
                for i, d in enumerate(self.state[agent]["observation"]["dice"]):
                    if need <= 0: break
                    if d == dtype or d == 8:
                        self.state[agent]["observation"]["dice"][i] = 0
                        need -= 1
            else:  # "any dice" requirement
                need = amt
                for i, d in enumerate(self.state[agent]["observation"]["dice"]):
                    if need <= 0: break
                    if d != 0:
                        self.state[agent]["observation"]["dice"][i] = 0
                        need -= 1
        
        # --- apply action ---
        if action_word == "end_round_action":
            self.state[agent]["observation"]["declared_end"] = 1
            # penalty for ending without using any dice that turn
            if all(d != 0 for d in self.state[agent]["observation"]["dice"]):  # all dice untouched
                self.rewards[agent] -= 100  # big penalty

        elif "card_type" not in self.actions[action_word]: 
            # character attack
            self.rewards[agent] += 10
            atk = self.actions[action_word]["dmg"]
            atk += self.state[agent]["observation"]["Kaeya"]["atk_permanent"]

            bonus = self.state[agent]["observation"]["Kaeya"]["atk_per_turn_amt"]
            used = self.state[agent]["observation"]["Kaeya"]["atk_per_turn_used"]
            if used == 0:
                self.state[agent]["observation"]["Kaeya"]["atk_per_turn_used"] = 1
                atk += bonus

            total_dmg = atk

            # energy handling
            if "energy" in self.actions[action_word]:
                self.state[agent]["observation"]["Kaeya"]["energy"] += self.actions[action_word]["energy"]
                max_e = self.state[agent]["observation"]["Kaeya"]["max_energy"]
                if self.state[agent]["observation"]["Kaeya"]["energy"] > max_e:
                    self.state[agent]["observation"]["Kaeya"]["energy"] = max_e
            elif self.state[agent]["observation"]["Kaeya"]["energy"] == self.state[agent]["observation"]["Kaeya"]["max_energy"]:
                # using burst consumes energy
                self.state[agent]["observation"]["Kaeya"]["energy"] = 0

            # deal damage
            self.state[other_agent]["observation"]["Kaeya"]["hp"] -= total_dmg
            if self.state[other_agent]["observation"]["Kaeya"]["hp"] < 0:
                self.state[other_agent]["observation"]["Kaeya"]["hp"] = 0

        else:
            # support card played
            self.rewards[agent] += 5
            # remove the card from hand (keep fixed size with trailing 0)
            cards = self.state[agent]["observation"]["cards"]
            w_id = self.word_to_id[action_word]
            if np.any(cards == w_id):
                idx = np.where(cards == w_id)[0][0]
                cards = np.delete(cards, idx)
            if len(cards) < 5:
                cards = np.append(cards, 0)
            self.state[agent]["observation"]["cards"] = cards

            # apply effects
            if "hp" in self.actions[action_word]:
                cur_hp = self.state[agent]["observation"]["Kaeya"]["hp"]
                cur_hp += self.actions[action_word]["hp"]
                max_hp = self.state[agent]["observation"]["Kaeya"]["max_hp"]
                self.state[agent]["observation"]["Kaeya"]["hp"] = min(max_hp, cur_hp)
                self.state[agent]["observation"]["Kaeya"]["full"] = 1

            if "atk_permanent" in self.actions[action_word]:
                self.state[agent]["observation"]["Kaeya"]["atk_permanent"] += self.actions[action_word]["atk_permanent"]
            if "atk_per_turn" in self.actions[action_word]:
                self.state[agent]["observation"]["Kaeya"]["atk_per_turn_amt"] = self.actions[action_word]["atk_per_turn"]
                self.state[agent]["observation"]["Kaeya"]["atk_per_turn_used"] = 0
            if "atk_discount" in self.actions[action_word]:
                self.state[agent]["observation"]["Kaeya"]["atk_discount"] = self.actions[action_word]["atk_discount"]

            if self.actions[action_word].get("card_type") == "artifact":
                self.state[agent]["observation"]["Kaeya"]["artifact"] = w_id
            if self.actions[action_word].get("card_type") == "weapon":
                self.state[agent]["observation"]["Kaeya"]["weapon"] = w_id

        # shaping reward: attack damage + retain HP
        if action in ["kaeya_normal", "kaeya_skill", "kaeya_burst"]:
            # add current hp to incentivize rounds that end with higher hp
            self.rewards[agent] += total_dmg + self.state[agent]["observation"]["Kaeya"]["hp"] 

        # kills opponent
        if self.state[other_agent]["observation"]["Kaeya"]["hp"] <= 0:
            self.rewards[agent] += 100              # big reward
            self.rewards[other_agent] -= 100        # big loss
            self.terminations[agent] = True 
            self.terminations[other_agent] = True
            self.infos[agent] = {"status": 'winner'}
            self.infos[other_agent] = {"status": 'loser'}

        # round transition if both have declared end
        if (
            self.state[agent]["observation"]["declared_end"]
            and self.state[other_agent]["observation"]["declared_end"]
        ):
            for p in [agent, other_agent]:
                self.state[p]["observation"]["Kaeya"]["atk_per_turn_used"] = 0
                self.state[p]["observation"]["Kaeya"]["full"] = 0
                self.state[p]["observation"]["declared_end"] = 0
                self.state[p]["observation"]["dice"] = np.array(self.generate_dice(self.dice_per_turn))
            self.turn += 1

        # end after 15 rounds
        if self.turn > 15:
            self.rewards[agent] -= 10
            self.rewards[other_agent] -= 10
            self.infos[agent] = {"status": "tie"}
            self.infos[other_agent] = {"status": "tie"}
            self.terminations[agent] = True
            self.terminations[other_agent] = True
            # return

        # accumulate rewards
        self._cumulative_rewards[agent] += self.rewards[agent]
        self._cumulative_rewards[other_agent] += self.rewards[other_agent]


        # switch player for the next step
        self.agent_selection = self._agent_selector.next()

    def close(self):
        pass