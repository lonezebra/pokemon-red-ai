"""
Prove the whole-game environment works before any training is launched
against it, and measure how fast it actually runs.

The throughput number is the point of this script as much as the checks are.
This architecture's viability rests entirely on steps-per-second: the policy
network is small and the GPU is not the constraint, PyBoy is, and PyBoy is
single-threaded CPU work. Measuring it decides whether this machine is
enough or whether the run needs to go somewhere with more cores.

    cd src && ../.venv/bin/python3 ../tools/smoke_test_whole_game.py
    cd src && ../.venv/bin/python3 ../tools/smoke_test_whole_game.py --steps 2000
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from envs.whole_game_env import (  # noqa: E402
    ACTIONS,
    FRAME_STACK,
    PokemonRedWholeGameEnv,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return bool(condition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=800)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    failures = []

    print("=" * 68)
    print(f"Whole-game env smoke test ({args.steps} random actions)")
    print("=" * 68)

    env = PokemonRedWholeGameEnv(max_steps=args.steps)

    try:
        print("\nSpaces")
        failures += [not check(
            "action space is one per button",
            env.action_space.n == len(ACTIONS),
            f"n={env.action_space.n}, actions={ACTIONS}",
        )]

        print("\nreset()")
        obs, info = env.reset()

        failures += [not check(
            "observation matches the declared space",
            env.observation_space.contains(obs),
        )]
        failures += [not check(
            "screen is the stacked, downsampled shape",
            obs["screen"].shape == (SCREEN_HEIGHT, SCREEN_WIDTH, FRAME_STACK)
            and obs["screen"].dtype == np.uint8,
            f"{obs['screen'].shape} {obs['screen'].dtype}",
        )]
        failures += [not check(
            "stats are finite and in 0-1",
            np.all(np.isfinite(obs["stats"]))
            and obs["stats"].min() >= 0.0 and obs["stats"].max() <= 1.0,
            f"{obs['stats']}",
        )]
        failures += [not check(
            "the screen is not blank",
            obs["screen"].std() > 0,
            f"std={obs['screen'].std():.2f}",
        )]

        print(f"\nRunning {args.steps} random steps...")
        totals = {}
        rewards = []
        start = time.monotonic()
        last_info = info

        # Counted across the whole run rather than sampled at the end. A
        # single check at the last step proves nothing either way: three
        # identical planes are the *correct* reading whenever the screen
        # happened to be static right then (walking into a wall renders no
        # animation at all), so the question worth asking is whether the
        # stack ever carries motion, not whether it does at one arbitrary
        # moment.
        steps_with_motion = 0

        for _ in range(args.steps):
            action = rng.integers(0, env.action_space.n)
            obs, reward, terminated, truncated, last_info = env.step(action)
            rewards.append(reward)
            for name, value in last_info["reward_components"].items():
                totals[name] = totals.get(name, 0.0) + value
            if not np.array_equal(obs["screen"][:, :, 0], obs["screen"][:, :, -1]):
                steps_with_motion += 1
            if terminated or truncated:
                break

        elapsed = time.monotonic() - start
        steps_per_second = len(rewards) / elapsed

        print("\nStep results")
        failures += [not check(
            "observation still matches the declared space",
            env.observation_space.contains(obs),
        )]
        failures += [not check(
            "every reward was finite",
            all(np.isfinite(r) for r in rewards),
        )]
        failures += [not check(
            "exploration actually fired -- random play found new tiles",
            last_info["tiles_explored"] > 1,
            f"{last_info['tiles_explored']} tiles",
        )]
        failures += [not check(
            "the frame stack carries motion, not three copies of one frame",
            steps_with_motion > 0,
            f"{steps_with_motion}/{len(rewards)} steps "
            f"({100.0 * steps_with_motion / max(len(rewards), 1):.0f}%) "
            f"showed a moving screen",
        )]

        print("\nReward components over the run")
        for name in sorted(totals):
            print(f"  {name:<10} {totals[name]:+9.3f}")
        print(f"  {'TOTAL':<10} {sum(rewards):+9.3f}")

        print("\nThroughput (single environment)")
        print(f"  {steps_per_second:.1f} agent-steps/sec")
        print(f"  {steps_per_second * 24:.0f} emulated frames/sec "
              f"({steps_per_second * 24 / 60:.1f}x real-time)")

        cores = os.cpu_count() or 1
        for envs in (cores - 4, cores):
            projected = steps_per_second * envs
            print(f"\n  Projected at {envs} parallel envs: "
                  f"{projected:.0f} steps/sec")
            for label, target in (("1M steps", 1_000_000),
                                  ("10M steps", 10_000_000)):
                hours = target / projected / 3600
                print(f"    {label:<10} ~{hours:6.1f} h "
                      f"({hours / 24:.1f} days)")

    finally:
        env.close()

    print()
    print("=" * 68)
    failed = sum(1 for f in failures if f)
    if failed:
        print(f"{failed} FAILURE(S)")
        print("=" * 68)
        return 1
    print("All checks passed")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
