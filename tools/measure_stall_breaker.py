"""
Measure how much of an episode the Oak's Lab stall is costing, by running
the same policy with and without a stall-breaker and comparing.

Instrumentation established that the 330M policy parks on Oak's Lab tile
(6,11) -- one tile east of the exit mat at (5,11) -- and presses DOWN into
the bottom wall: 11,844 such presses across four episodes, zero of which
moved the player, while every real exit happened from (5,11). A scripted
LEFT-then-DOWN escaped 4 times out of 4. Roughly half of every episode was
being spent this way.

The breaker here is a gymnasium Wrapper, deliberately NOT a change to
whole_game_env.py: the training environment stays exactly as the 330M policy
was trained against, so this can never contaminate a future run, and the
comparison is the same policy against two environments that differ in one
rule.

That rule: if the player's tile has not changed for STALL_LIMIT consecutive
steps and the game is not in battle, override the policy's action with a
uniform random one until the player moves again. Battles are excluded
because position is frozen there by design and mashing random buttons
through a fight would measure something other than what this is asking.

This is scaffolding, not learning, and it is worth being explicit that it
does not fix the policy -- the policy still cannot tell (5,11) from (6,11).
It exists to price the bug: if unsticking recovers most of the lost episode,
that justifies paying for the real fix (position in the observation, retried
with the optimizer state carried across). If it recovers little, the stall
was a symptom and the real problem is elsewhere.

    cd src && ../.venv/bin/python3 ../tools/measure_stall_breaker.py
    cd src && ../.venv/bin/python3 ../tools/measure_stall_breaker.py --episodes 8
"""

import argparse
import os
import random
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from core.memory import is_in_battle  # noqa: E402
from envs.whole_game_env import ACTIONS, PokemonRedWholeGameEnv  # noqa: E402
from watch_whole_game import resolve_model  # noqa: E402

OAKS_LAB = 40


DIRECTIONS = frozenset(
    ACTIONS.index(name) for name in ("up", "down", "left", "right")
)


class StallBreaker(gym.Wrapper):
    """Force random actions once the player has walked into a wall too long.

    Counts *walking into a wall*, not merely standing still: a step only
    counts toward the stall if the action was a direction AND the player's
    tile did not change. That distinction is what lets the limit come down.
    "Position unchanged" alone is also true throughout dialogue -- where the
    player is frozen by design and the correct action is to press A -- so a
    tight limit on that signal would fire on every text box and mash random
    buttons through the intro, Oak's speech, and the delivery cutscene. A
    direction pressed with no movement means a wall (or an NPC), which is
    exactly the state worth interrupting.

    Battles are excluded for the same reason: position is meaningless there.
    """

    def __init__(self, env, limit):
        super().__init__(env)
        self.limit = limit
        self._prev = None
        self._stall = 0
        self.forced_actions = 0
        self.stalls_broken = 0
        self.longest_stall = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = None
        self._stall = 0
        return obs, info

    def step(self, action):
        breaking = self._stall >= self.limit
        if breaking:
            # Directions only. A random pick from the full set would spend
            # a third of its attempts on A/B, which cannot move the player
            # off a wall and can open menus that make things worse.
            action = random.choice(tuple(DIRECTIONS))
            self.forced_actions += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = (info["map_id"], info["x"], info["y"])
        moved = pos != self._prev

        walked_into_wall = (
            not moved
            and int(action) in DIRECTIONS
            and not is_in_battle(self.env.unwrapped.pyboy)
        )
        if walked_into_wall:
            self._stall += 1
            self.longest_stall = max(self.longest_stall, self._stall)
        else:
            if breaking and moved:
                self.stalls_broken += 1
            self._stall = 0
        self._prev = pos
        return obs, reward, terminated, truncated, info


def play(model, env, max_steps):
    obs, _ = env.reset()
    prev = None
    delivered = False
    delivery_step = None
    frozen = 0
    lab_steps = 0
    lab_tile_counts = Counter()
    total_reward = 0.0
    info = {}

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        pos = (info["map_id"], info["x"], info["y"])

        if info["reward_components"].get("milestone", 0.0) >= 100.0 and not delivered:
            delivered = True
            delivery_step = step

        if info["map_id"] == OAKS_LAB:
            lab_steps += 1
            lab_tile_counts[(info["x"], info["y"])] += 1
        if pos == prev:
            frozen += 1

        prev = pos
        if terminated or truncated:
            break

    steps = step + 1
    return {
        "steps": steps,
        "delivered": delivered,
        "delivery_step": delivery_step,
        "frozen_pct": 100.0 * frozen / steps,
        "lab_pct": 100.0 * lab_steps / steps,
        "stuck_tile_steps": lab_tile_counts[(6, 11)],
        "tiles": info.get("tiles_explored", 0),
        "events": info.get("events", 0),
        "badges": info.get("badges", 0),
        "reward": total_reward,
    }


def summarise(label, rows, extra=""):
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    delivered = sum(r["delivered"] for r in rows)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  delivered            {delivered}/{n}")
    print(f"  mean tiles explored  {mean('tiles'):.0f}")
    print(f"  mean events          {mean('events'):.1f}")
    print(f"  mean reward          {mean('reward'):+.1f}")
    print(f"  mean % steps frozen  {mean('frozen_pct'):.1f}%")
    print(f"  mean % steps in lab  {mean('lab_pct'):.1f}%")
    print(f"  mean steps on (6,11) {mean('stuck_tile_steps'):.0f}")
    print(f"  badges (best)        {max(r['badges'] for r in rows)}")
    if extra:
        print(f"  {extra}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--limits", default="5,10,25,50",
                        help="comma-separated wall-press counts to compare; "
                             "each is run as its own arm against one baseline")
    args = parser.parse_args()

    limits = [int(x) for x in args.limits.split(",") if x.strip()]
    model_path = resolve_model(None)
    print(f"Policy: {model_path.name}")
    print(f"Sweeping limits: {limits} (wall-presses, battles excluded)")
    model = PPO.load(model_path, device="cpu")

    base_env = PokemonRedWholeGameEnv(max_steps=args.max_steps)
    arms = {}
    try:
        print(f"\nBaseline, no breaker ({args.episodes} episodes)...", flush=True)
        before = [play(model, base_env, args.max_steps) for _ in range(args.episodes)]

        for limit in limits:
            broken = StallBreaker(base_env, limit=limit)
            print(f"Limit {limit} ({args.episodes} episodes)...", flush=True)
            rows = [play(model, broken, args.max_steps) for _ in range(args.episodes)]
            arms[limit] = (rows, broken)
    finally:
        base_env.close()

    summarise("Baseline (no breaker)", before)
    for limit, (rows, broken) in arms.items():
        summarise(
            f"Limit {limit}", rows,
            extra=f"forced actions {broken.forced_actions}, "
                  f"stalls broken {broken.stalls_broken}, "
                  f"longest wall-press run {broken.longest_stall}",
        )

    mean = lambda rows, k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    print("\nComparison")
    print("-" * 74)
    print(f"{'limit':>7}{'delivered':>12}{'tiles':>9}{'events':>9}"
          f"{'reward':>10}{'% in lab':>10}{'(6,11)':>9}")
    print(f"{'none':>7}{sum(r['delivered'] for r in before):>7}/{len(before):<4}"
          f"{mean(before,'tiles'):>9.0f}{mean(before,'events'):>9.1f}"
          f"{mean(before,'reward'):>10.1f}{mean(before,'lab_pct'):>9.1f}%"
          f"{mean(before,'stuck_tile_steps'):>9.0f}")
    for limit, (rows, _) in arms.items():
        print(f"{limit:>7}{sum(r['delivered'] for r in rows):>7}/{len(rows):<4}"
              f"{mean(rows,'tiles'):>9.0f}{mean(rows,'events'):>9.1f}"
              f"{mean(rows,'reward'):>10.1f}{mean(rows,'lab_pct'):>9.1f}%"
              f"{mean(rows,'stuck_tile_steps'):>9.0f}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
