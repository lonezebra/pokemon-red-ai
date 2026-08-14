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
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from core.memory import is_in_battle  # noqa: E402
from envs.whole_game_env import PokemonRedWholeGameEnv  # noqa: E402
from watch_whole_game import resolve_model  # noqa: E402

OAKS_LAB = 40


class StallBreaker(gym.Wrapper):
    """Force random actions once the player has been immobile too long."""

    def __init__(self, env, limit):
        super().__init__(env)
        self.limit = limit
        self._prev = None
        self._stall = 0
        self.forced_actions = 0
        self.stalls_broken = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = None
        self._stall = 0
        return obs, info

    def step(self, action):
        breaking = self._stall >= self.limit
        if breaking:
            action = self.action_space.sample()
            self.forced_actions += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = (info["map_id"], info["x"], info["y"])

        if pos == self._prev and not is_in_battle(self.env.unwrapped.pyboy):
            self._stall += 1
        else:
            if breaking and pos != self._prev:
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
    parser.add_argument("--limit", type=int, default=50,
                        help="consecutive immobile steps before breaking in")
    args = parser.parse_args()

    model_path = resolve_model(None)
    print(f"Policy: {model_path.name}")
    print(f"Stall limit: {args.limit} immobile steps (battles excluded)")
    model = PPO.load(model_path, device="cpu")

    base_env = PokemonRedWholeGameEnv(max_steps=args.max_steps)
    try:
        print(f"\nRunning {args.episodes} episodes WITHOUT the breaker...", flush=True)
        before = [play(model, base_env, args.max_steps) for _ in range(args.episodes)]

        broken = StallBreaker(base_env, limit=args.limit)
        print(f"Running {args.episodes} episodes WITH the breaker...", flush=True)
        after = [play(model, broken, args.max_steps) for _ in range(args.episodes)]
    finally:
        base_env.close()

    summarise("WITHOUT stall-breaker (baseline)", before)
    summarise(
        "WITH stall-breaker", after,
        extra=f"forced actions {broken.forced_actions}, "
              f"stalls broken {broken.stalls_broken}",
    )

    n = len(before)
    b_tiles = sum(r["tiles"] for r in before) / n
    a_tiles = sum(r["tiles"] for r in after) / n
    b_lab = sum(r["lab_pct"] for r in before) / n
    a_lab = sum(r["lab_pct"] for r in after) / n
    print("\nDelta")
    print("-----")
    print(f"  tiles explored   {b_tiles:.0f} -> {a_tiles:.0f} "
          f"({100*(a_tiles-b_tiles)/max(1,b_tiles):+.0f}%)")
    print(f"  % steps in lab   {b_lab:.1f}% -> {a_lab:.1f}%")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
