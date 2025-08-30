'''
Training a PPO agent. 
Must select the env_mode (discrete/random) and ppo_mode (masked/unmasked).
'''

import os, sys, time, math, argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import numpy as np
from collections import Counter

import gymnasium as gym
from gymnasium import spaces
import supersuit as ss

from pettingzoo import AECEnv
from pettingzoo.utils import parallel_to_aec, wrappers, aec_to_parallel
from pettingzoo.utils.wrappers import BaseParallelWrapper

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

import wandb
from wandb.integration.sb3 import WandbCallback

env_mode = "random" # "random" or "discrete"
if env_mode == "random":
    from env_mini_random.mini_eval_utils import GymEnv
    from env_mini_random.mini_eval_utils import eval_model_vs_valid_random, show_one_game, play_one_game
elif env_mode == "discrete":
    from env_mini_discrete.mini_eval_utils import GymEnv
    from env_mini_discrete.mini_eval_utils import eval_model_vs_valid_random, show_one_game, play_one_game

ppo_mode = "masked" # "masked" or "unmasked"

# --- Turn AEC env into a GymEnv ------------------------------------------------
def make_env():
    env = GymEnv()                                      # AEC env
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    if ppo_mode == "unmasked":
        env = aec_to_parallel(env)                      # parallel env (for Supersuit)
        env = ss.flatten_v0(env)                        # flatten env (for MlpPolicy)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
        env = VecMonitor(env)                           # optional for episode stats
    if ppo_mode == "masked":
        env = AECToGymWrapper(env)
        env = ActionMasker(env, mask_fn)
    return env
class AECToGymWrapper(gym.Env):
    def __init__(self, aec_env, debug=False):
        super().__init__()
        self.debug = debug
        self.aec_env = aec_env
        self.aec_env.reset()

        cur = self.aec_env.agent_selection
        self.observation_space = self.aec_env.observation_space(cur)
        self.action_space      = self.aec_env.action_space(cur)

        self.agent = cur                     # acting agent pointer
        self.valid_action_mask = self.get_action_masks()  # must correspond to self.agent

    # ---------- tiny helper ----------
    def _assert_sync(self, where):
        assert self.agent == self.aec_env.agent_selection, (
            f"[{where}] desync: wrapper.agent={self.agent} "
            f"env.agent_selection={self.aec_env.agent_selection}"
        )

    def _print_mask(self, where):
        if not self.debug: return
        m = self.get_action_masks().astype(int)
        vocab = getattr(self.aec_env, "vocab", [str(i) for i in range(len(m))])
        print(f"\n[{where}] agent turn: {self.agent}")
        print("mask:", m.tolist())
        for i, ok in enumerate(m):
            print(f"{i:2d} {vocab[i]:18s} legal={ok}")

    # ---------- gym API ----------
    def reset(self, **kwargs):
        self.aec_env.reset()
        self.agent = self.aec_env.agent_selection       # sync to current
        self._assert_sync("reset-after-sync")
        obs = self.aec_env.observe(self.agent)
        self.valid_action_mask = self.get_action_masks()
        self._print_mask("reset")
        return obs, {}

    def step(self, action):
        # ensure we are choosing for the current agent
        self._assert_sync("pre-step (acting)")
        acting = self.agent

        if self.debug:
            print(f"\n[pre-step] acting={acting} action_idx={action}")

        # step env (advances AEC internal pointer)
        self.aec_env.step(action)

        # collect results for the agent that JUST acted
        terminated = bool(self.aec_env.terminations[acting])
        truncated  = bool(self.aec_env.truncations[acting])
        reward     = float(self.aec_env.rewards[acting])

        # advance our pointer to whoever is up next
        self.agent = self.aec_env.agent_selection
        self._assert_sync("post-step (next)")
        obs = None if (terminated or truncated) else self.aec_env.observe(self.agent)

        # refresh mask for the NEXT agent (this obs)
        self.valid_action_mask = self.get_action_masks()
        self._print_mask("post-step")
        return obs, reward, terminated, truncated, {}

    def get_action_masks(self):
        # extra guard: always serve mask for the current agent
        self._assert_sync("get_action_masks")
        m = self.aec_env.get_action_mask(self.agent)
        return np.asarray(m, dtype=bool)

    def render(self): return self.aec_env.render()
    def close(self):  self.aec_env.close()

def mask_fn(env):     # used by ActionMasker
    return env.valid_action_mask


# --- Custom evaluation vs random bot -------------------------------------------
class AECPvRandomEvalCallback(BaseCallback):
    """
    Periodically evaluates current policy vs valid-random using the *AEC env*,
    and logs to Weights & Biases.

    Args:
      side: "0" or "1" — which agent SB3 controls during eval
      eval_freq: run evaluation every N training steps
      n_games: number of eval episodes per evaluation
      seed: base seed for reproducibility (optional)
      wandb_prefix: prefix for metric keys in W&B
    """
    def __init__(self, side="0", eval_freq=10_000, n_games=20, seed=None, wandb_prefix="eval", verbose=0):
        super().__init__(verbose)
        assert side in ("0", "1")
        self.side = side
        self.other = "1" if side == "0" else "0"
        self.eval_freq = int(eval_freq)
        self.n_games = int(n_games)
        self.seed = seed
        self.prefix = wandb_prefix

    def _log_wandb(self, d: dict, commit: bool):
        try:
            import wandb
            wandb.log(d, step=self.num_timesteps, commit=commit)
        except Exception:
            pass

    def _on_step(self) -> bool:
        # run eval every eval_freq calls
        if self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0):
            wins = Counter()
            # per-episode logging
            for i in range(self.n_games):
                # offset seed per episode so results are stable/reproducible
                ep_seed = None if self.seed is None else (self.seed + self.n_calls + i)
                if ppo_mode == True:
                    result, _ = play_one_game(
                        model=self.model,
                        side=self.side,
                        deterministic=True,
                        seed=ep_seed,
                        verbose=False,
                        masked=True
                    )
                else:
                    result, _ = play_one_game(
                        model=self.model,
                        side=self.side,
                        deterministic=True,
                        seed=ep_seed,
                        verbose=False,
                        masked=False
                    )
                wins[result] += 1

                # log a per-episode flag set
                self._log_wandb({
                    f"{self.prefix}/episode_win_sb3": int(result == self.side),
                    f"{self.prefix}/episode_win_rand": int(result == self.other),
                    f"{self.prefix}/episode_tie": int(result == "tie"),
                    f"{self.prefix}/sb3_player": int(self.side),
                    f"{self.prefix}/game_idx": i,
                }, commit=False)

            total = max(1, self.n_games)
            agg = {
                f"{self.prefix}/win_rate_sb3": wins[self.side] / total,
                f"{self.prefix}/win_rate_rand": wins[self.other] / total,
                f"{self.prefix}/tie_rate": wins["tie"] / total,
                f"{self.prefix}/p0_win_rate": wins["0"] / total,
                f"{self.prefix}/p1_win_rate": wins["1"] / total,
                f"{self.prefix}/games": total,
                f"{self.prefix}/sb3_player": int(self.side),
            }
            # one aggregate commit per eval block
            self._log_wandb(agg, commit=True)

            if self.verbose:
                print("[AEC Eval]", agg)

        return True
# -------------------------------------------------------------------------------

# main script
if __name__ == "__main__":
    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 1_000_000, # 250000
        # "total_timesteps": 250000,
        "env_name": GymEnv.__name__,
    }
    run = wandb.init(
        project="mini_ppo_training",
        config=config,
        sync_tensorboard=True,          # auto-upload sb3's tensorboard metrics
        save_code=True,  
    )

    env = make_env()
    obs = env.reset()

    if ppo_mode == "unmasked":
        model = PPO(
            policy=config["policy_type"],
            env=env,
            verbose=1,
            tensorboard_log=f"runs/{run.id}",
            n_steps=128,                    
            batch_size=128,
        )
    if ppo_mode == "masked":
        model = MaskablePPO(
            policy=config["policy_type"],
            env=env,     # action masked env
            n_steps=128,           
            batch_size=32,
            normalize_advantage=True,
            verbose=1
        )
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            AECPvRandomEvalCallback(
                side="0",               # set to 0 or 1
                eval_freq=10_000,       # evaluate every 10k env steps
                n_games=100, 
                verbose=1
            )
        ],
        progress_bar=True,
    )

    model.save("models/ppo_mini_" + env_mode + "_" + ppo_mode)
    run.finish()
