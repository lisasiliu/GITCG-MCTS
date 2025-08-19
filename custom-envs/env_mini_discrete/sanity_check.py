# -------------------------------
# Sanity checks & quick eval
# Paste BELOW the env class
# -------------------------------
import random
from collections import defaultdict, Counter
from env_mini_discrete import GITCGDiscreteMiniGymEnv
import numpy as np

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

def _fmt_step_line(round_no, agent, action_word, hp_self, hp_opp,
                   dice_before, dice_after, hand_fixed, de_self, de_opp):
    # Choose fixed widths so '|' columns align nicely.
    # Tune widths here if you want more/less room.
    return (
        f"Round {round_no:<2} | "
        f"Agent {agent} | "
        f"{action_word:<18} | "               # action column (16 chars)
        f"HP s={hp_self:>2} o={hp_opp:>2} | " # hp column
        f"Dice {dice_before:>2}->{dice_after:<2} | "
        f"Hand: {hand_fixed} | "
        f"DE s={de_self} o={de_opp}"
    )

def _masked_random_action(mask):
    """Uniformly sample from ALL valid actions (including end_round_action)."""
    valid_idxs = [i for i, v in enumerate(mask) if v == 1]
    if not valid_idxs:
        # Fallback to end_round if somehow nothing is valid
        return env.word_to_id["end_round_action"] - 1
    return np.random.choice(valid_idxs)

def play_one_game(env, seed=None, verbose=False):
    """Random-vs-Random; returns winner: '0', '1', or 'tie'."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env.reset()

    # Track per-round attack counts to catch impossible sequences
    # (With 4 dice/round and any-action-ends-turn, max 2 attacks in a single round
    # only if an attack discount is active; otherwise max 1.)
    per_round_attacks = { "0": Counter(), "1": Counter() }  # key = env.turn, value = count in that round
    # Track whether we've printed this agent's end-round in the current round
    printed_end_this_round = {"0": -1, "1": -1}  # stores the round number for which end was printed


    # Quick helpers to peek at internal state (your fixed env keeps runtime state in env.state)
    def dice(agent): return env.state[agent]["observation"]["dice"]
    def kaeya(agent): return env.state[agent]["observation"]["Kaeya"]

    # Step loop
    safety_step_cap = 2000  # extra safety; game usually ends much earlier
    steps = 0
    last_turn_seen = env.turn

    while not all(env.terminations.values()) and steps < safety_step_cap:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last(observe=True)

        # 1) Mask-respecting random action
        mask = env.get_action_mask(agent)
        action = _masked_random_action(mask)

        # 2) Invariant: never pick an invalid action
        assert mask[action] == 1 or sum(mask) == 0, f"Chose invalid action for agent {agent}"

        # 3) Snapshot dice before step for post-check
        dice_before = dice(agent)

        # 4) Name for logging & attack counting
        action_word = env.id_to_word[action + 1]

        # 4a) Track whether end is declared to only print end round action once
        opp = "0" if agent == "1" else "1"
        pre_turn = env.turn
        pre_de_self = env.state[agent]["observation"]["declared_end"]
        pre_de_opp  = env.state[opp]["observation"]["declared_end"]
        dice_before = dice(agent)  
        
        env.step(action)

        # 6) Dice should not go negative
        assert dice(agent) >= 0, f"Negative dice after action {action_word} by agent {agent}"

        # 7) Per-round attack counting (only for kaeya_* attacks)
        if action_word in ("kaeya_normal", "kaeya_skill", "kaeya_burst"):
            per_round_attacks[agent][env.turn] += 1

        # 8) Sanity: you cannot do *three* attacks in the same round
        # (Given 4 dice/round, and any action ends turn, 3 attacks in one round is impossible even with a -1 discount.)
        if per_round_attacks[agent][env.turn] > 2:
            raise AssertionError(
                f"Agent {agent} performed >2 attacks in round {env.turn} (impossible under rules)"
            )

        # Did the round advance inside this step?
        post_turn = env.turn
        advanced = (post_turn != pre_turn)

        # Decide whether to print this step
        should_print = True
        if action_word == "end_round_action":
            # Only print the FIRST time this agent declares end in THIS round
            # i.e., when it transitions from 0 -> 1, and we haven't printed it yet for env.turn
            if not (pre_de_self == 0 and printed_end_this_round[agent] != env.turn):
                should_print = False
            else:
                printed_end_this_round[agent] = env.turn  # remember we've printed in this round

        # Compute what the declared_end *logically* is right after this action,
        # i.e., before the env's round-advance reset can zero them out.
        # - For non-end actions: flags don't change.
        # - For end_round_action: the acting agent flips 0->1.
        log_de_self = pre_de_self
        log_de_opp  = pre_de_opp
        if action_word == "end_round_action" and pre_de_self == 0:
            log_de_self = 1

        # If the round advanced, that means both ends are now in effect.
        if advanced:
            log_de_self = 1
            log_de_opp  = 1

        # Build an aligned line that shows the *pre-step* round number,
        # and (optionally) an advance tag.
        adv_tag = f" | ADV→R{post_turn}" if advanced else ""

        # NOTE: use the pre-step round number for display so the action
        # is attributed to the round it actually belonged to.
        if verbose and should_print:
            hand_fixed = hand_str(env, agent)
            line = _fmt_step_line(
                round_no=pre_turn,                    # <-- show pre-step round
                agent=agent,
                action_word=action_word,
                hp_self=kaeya(agent)["hp"],
                hp_opp=kaeya(opp)["hp"],
                dice_before=dice_before,
                dice_after=dice(agent),               # may show reset; see note below
                hand_fixed=hand_fixed,
                de_self=log_de_self,                  # <-- logical flags after action
                de_opp=log_de_opp
            )
            print(line + adv_tag)

        steps += 1

        # Early break on full termination (env already sets terminations both sides on KO)
        if all(env.terminations.values()):
            break

    # Determine result
    if env.infos.get("0", {}).get("status") == "winner":
        return "0"
    if env.infos.get("1", {}).get("status") == "winner":
        return "1"
    return "tie"

def eval_random_vs_random(n_games=100, seed=0, verbose_every=None):
    """Run N random self-play games; return a small dict with win/tie rates."""
    wins = Counter()
    for i in range(n_games):
        # fresh env each game to avoid any hidden carry-over
        e = GITCGDiscreteMiniGymEnv()
        result = play_one_game(e, seed=seed + i, verbose=False)
        wins[result] += 1
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"[{i+1}/{n_games}] so far: {dict(wins)}")

    stats = {
        "games": n_games,
        "p0_win_rate": wins["0"] / n_games,
        "p1_win_rate": wins["1"] / n_games,
        "tie_rate":   wins["tie"] / n_games,
        "raw_counts": dict(wins),
    }
    return stats

# -------------------------------
# CLI / quick run
# -------------------------------
if __name__ == "__main__":
    # Quick smoke test: 100 games random vs. random
    env = GITCGDiscreteMiniGymEnv()
    print("Running 100 random self-play sanity games...")
    stats = eval_random_vs_random(n_games=100, seed=42)
    print("Results:", stats)

    # A couple of invariant spot-checks on a single verbose game
    print("\nRunning one verbose game to eyeball actions...")
    _ = play_one_game(GITCGDiscreteMiniGymEnv(), seed=123, verbose=True)
    print("Done.")
