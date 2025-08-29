import random
from collections import Counter
import numpy as np
from env_mini_random.env_mini_random import GITCGRandomMiniGymEnv as GymEnv

# ----------- Pretty print helper functions --------------------------------------

def hand_str(env, agent, token_width: int = 5, list_width: int = 28) -> str:
    """
    Pretty + fixed-width hand string.
    Example: [broke, hash_, sweet, sweet, skywa]  (padded/truncated to list_width)
    """
    cards = env.state[agent]["observation"]["cards"]
    toks = [env.id_to_word[int(cid)][:token_width] for cid in cards if int(cid) != 0]
    s = "[" + ", ".join(toks) + "]"
    # pad/truncate to stabilize the column width
    if len(s) < list_width:
        s = s + " " * (list_width - len(s))
    else:
        s = s[:list_width]
    return s

def dice_str(dice_nums):
    '''
    List of dice numbers. This will be converted to a string list.
    Empty (0) dice are skipped.
    Example: [7 3 0 0] -> [cryo anemo]
    '''
    dice_type = ["hydro", "pyro", "anemo", "geo", "dendro", "electro", "cryo", "omni"]
    dice_to_id = {word: idx for idx, word in enumerate(dice_type, start=1)}
    id_to_dice = {idx: word for word, idx in dice_to_id.items()}
    return '[' + ' '.join([id_to_dice[d] for d in dice_nums if d != 0]) + ']'

def _fmt_step_line(round_no, agent, action_word, hp_self, hp_opp,
                   dice_before, dice_after, dice_before_actual, dice_after_actual, hand_fixed, de_self, de_opp):
    '''
    Formatting each step line.
    Example: Round 4  | Agent 0 | hash_brown         | HP s= 7 o=10 | Dice  4->3  | Hand: [broke]   
    '''
    # Choose fixed widths so '|' columns align nicely.
    return (
        f"Round {round_no:<2} | "
        f"Agent {agent} | "
        f"{action_word:<18} | "               # action column (16 chars)
        f"HP s={hp_self:>2} o={hp_opp:>2} | " # hp column
        f"Dice {dice_before:>2}->{dice_after:<2} -- {dice_str(dice_before_actual) + "->" + dice_str(dice_after_actual):<60}| "
        f"Hand: {hand_fixed} | "
        f"DE s={de_self} o={de_opp}"
    )


# ---------- masked random must accept env or compute fallback cleanly ----------
def _masked_random_action(mask, env=None):
    """Uniformly sample from valid actions in mask (1=valid).
       Falls back to end_round_action if nothing valid.
    """
    valid_idxs = [i for i, v in enumerate(mask) if v == 1]
    if valid_idxs:
        return int(np.random.choice(valid_idxs))
    # fallback
    if env is not None:
        return env.word_to_id["end_round_action"] - 1
    return 0  # safe default if env not provided


# ---------- Helper: robust winner detection ----------
def _winner_from_env(env) -> str:
    """Return '0' | '1' | 'tie' using env.infos when possible, else HP fallback."""
    # Prefer infos dict
    info0 = env.infos.get("0")
    info1 = env.infos.get("1")
    if isinstance(info0, dict) and isinstance(info1, dict):
        s0 = info0.get("status")
        s1 = info1.get("status")
        if s0 == "winner" and s1 == "loser":
            return "0"
        if s1 == "winner" and s0 == "loser":
            return "1"
        if s0 == "tie" and s1 == "tie":
            return "tie"

    # Fallback: decide from HP
    hp0 = env.state["0"]["observation"]["Kaeya"]["hp"]
    hp1 = env.state["1"]["observation"]["Kaeya"]["hp"]
    if hp0 > 0 and hp1 <= 0:
        return "0"
    if hp1 > 0 and hp0 <= 0:
        return "1"
    return "tie"


# ---------- Core: play one game (policy vs valid-random) and optionally pretty print ----------
def play_one_game(model, side="0", deterministic=True, seed=None, verbose=False, masked=False):
    """
    Plays a single AEC game: SB3 policy on `side` ('0' or '1') vs valid-random opponent.
    If `verbose`, pretty prints each (first) end-round action + every non-end action.
    Returns: ('0' | '1' | 'tie'), plus the collected lines if you want them.
    """
    assert side in ("0", "1"), "side must be '0' or '1'"
    other = "1" if side == "0" else "0"

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = GymEnv()
    env.reset(seed=seed)

    # track prints: only show each agent's first end-declare per round
    printed_end_this_round = {"0": -1, "1": -1}
    lines = []

    def dice(agent):  return env.state[agent]["observation"]["dice"]
    def kaeya(agent): return env.state[agent]["observation"]["Kaeya"]

    safety_step_cap = 2000
    steps = 0

    while not all(env.terminations.values()) and steps < safety_step_cap:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last(observe=True)

        mask = env.get_action_mask(agent)

        # pick action
        if agent == side:
            if masked == True:
                act, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
            else:
                act, _ = model.predict(obs, deterministic=deterministic)
            act = int(act)
            if mask[act] != 1:
                # if policy chose invalid, replace with valid random (or end)
                act = _masked_random_action(mask, env)
        else:
            act = _masked_random_action(mask, env)

        # snapshot (pre-step) for logging
        opp = "0" if agent == "1" else "1"
        pre_turn    = env.turn
        pre_de_self = env.state[agent]["observation"]["declared_end"]
        pre_de_opp  = env.state[opp]["observation"]["declared_end"]
        dice_before = dice(agent)

        action_word = env.id_to_word[act + 1]

        # step
        env.step(act)

        # detect round advance
        post_turn = env.turn
        advanced = (post_turn != pre_turn)

        # logical declared_end state immediately after this action (before resets)
        log_de_self = pre_de_self
        log_de_opp  = pre_de_opp
        if action_word == "end_round_action" and pre_de_self == 0:
            log_de_self = 1
        if advanced:
            log_de_self = 1
            log_de_opp  = 1

        # only print the first "end_round_action" per agent per round
        should_print = True
        if action_word == "end_round_action":
            if not (pre_de_self == 0 and printed_end_this_round[agent] != pre_turn):
                should_print = False
            else:
                printed_end_this_round[agent] = pre_turn

        if verbose and should_print:
            hand_fixed = hand_str(env, agent)
            line = _fmt_step_line(
                round_no=pre_turn,
                agent=agent,
                action_word=action_word,
                hp_self=kaeya(agent)["hp"],
                hp_opp=kaeya(opp)["hp"],
                dice_before=sum(d != 0 for d in dice_before),   # count of usable dice before
                dice_after=sum(d != 0 for d in dice(agent)),    # count of usable dice after
                dice_before_actual = dice_before,
                dice_after_actual = dice(agent), 
                hand_fixed=hand_fixed,
                de_self=log_de_self,
                de_opp=log_de_opp,
            )
            if advanced:
                line += f" | ADV→R{post_turn}"
            print(line)
            lines.append(line)

        steps += 1

        if all(env.terminations.values()):
            break

    result = _winner_from_env(env)
    return result, lines


# ---------- Evaluate many games; log per-episode wins for both sides ----------
def eval_model_vs_valid_random(model, side="0", n_games=100, seed=None, deterministic=True, log_wandb=False):
    """
    Runs `n_games` episodes: SB3 (as `side`) vs valid-random. Returns stats dict.
    If `log_wandb`, logs per-episode win flags and aggregate rates.
    """
    assert side in ("0", "1")
    other = "1" if side == "0" else "0"

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    wins = Counter()

    for i in range(n_games):
        result, _ = play_one_game(
            model=model,
            side=side,
            deterministic=deterministic,
            seed=None if seed is None else seed + i,
            verbose=False,
        )
        wins[result] += 1

        if log_wandb:
            try:
                import wandb
                wandb.log({
                    "eval/episode_win_sb3": int(result == side),
                    "eval/episode_win_rand": int(result == other),
                    "eval/episode_tie": int(result == "tie"),
                    "eval/sb3_player": int(side),
                    "eval/game_idx": i,
                })
            except Exception:
                pass

    total = max(1, n_games)
    stats = {
        "games": total,
        "sb3_player": side,
        "p0_win_rate": wins["0"] / total,
        "p1_win_rate": wins["1"] / total,
        "tie_rate":   wins["tie"] / total,
        "win_rate_sb3": wins[side] / total,
        "win_rate_rand": wins[other] / total,
        "raw_counts": dict(wins),
    }

    if log_wandb:
        try:
            import wandb
            wandb.log({
                "eval/win_rate_sb3": stats["win_rate_sb3"],
                "eval/win_rate_rand": stats["win_rate_rand"],
                "eval/tie_rate": stats["tie_rate"],
                "eval/sb3_player": int(side),
                "eval/games": total,
            })
        except Exception:
            pass

    return stats


# ---------- Convenience: show one random game (pretty) ----------
def show_one_game(model, side="0", seed=None, deterministic=True):
    """Plays 1 game with pretty printing and prints final result."""
    result, _ = play_one_game(
        model=model,
        side=side,
        deterministic=deterministic,
        seed=seed,
        verbose=True,
    )
    print(f"RESULT: winner = {result}\n")
    return result
