"""
Measure how often a *random* policy reaches the forest exit from each
curriculum checkpoint.

    .venv/bin/python3 tools/measure_curriculum_signal.py [--episodes N] [--max-steps N]

This is the empirical test of the premise behind curriculum training,
run before committing to it rather than after. The claim is that
entrance-start training fails because a success is a 127-move correct
sequence, so essentially every episode ends in failure and backward
replay spends its strength propagating failure; starting near the goal
should invert that ratio.

That claim is only worth acting on if success is actually reachable by
chance near the goal. If a random policy already exits often from d=10
there is real signal for the value function to latch onto and the
curriculum's first stage will train; if it exits ~never even from 10
hops out, the problem is not the distance to the goal and curriculum
training would inherit the same starvation it was meant to fix.

Reports per-checkpoint success rate and median steps-to-exit, deepest
first, so the falloff with distance is visible rather than inferred.
"""

import argparse
import pathlib
import random
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from actions import num_actions
from core.config import PROJECT_ROOT
from envs.forest_env import PokemonRedForestEnv

CURRICULUM_DIR = PROJECT_ROOT / "saves" / "forest_curriculum"


def checkpoint_states():
    """Curriculum states as (distance, path), nearest the goal first."""
    states = []
    for path in CURRICULUM_DIR.glob("d*.state"):
        states.append((int(path.name[1:4]), path))
    return sorted(states)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--distances", type=int, nargs="*",
                        help="only measure these checkpoint distances")
    args = parser.parse_args()

    states = checkpoint_states()
    if args.distances:
        wanted = set(args.distances)
        states = [(d, p) for d, p in states if d in wanted]
    if not states:
        print(f"no curriculum states in {CURRICULUM_DIR} -- run "
              f"tools/build_curriculum_states.py first")
        return 1

    print(f"{args.episodes} random episodes per checkpoint, "
          f"max {args.max_steps} steps each\n")
    print(f"{'dist':>5}  {'success':>8}  {'median steps':>13}  {'best depth':>11}")

    for distance, path in states:
        env = PokemonRedForestEnv(max_steps=args.max_steps, start_states=[path])
        successes = []
        best_depths = []
        for _ in range(args.episodes):
            env.reset()
            done = False
            info = {}
            while not done:
                _, _, done, info = env.step(random.randrange(num_actions()))
            if info.get("reached_goal"):
                successes.append(info["step_count"])
            best_depths.append(info.get("min_distance"))
        env.close()

        rate = 100.0 * len(successes) / args.episodes
        median = f"{statistics.median(successes):.0f}" if successes else "-"
        depths = [d for d in best_depths if d is not None]
        best = min(depths) if depths else None
        print(f"{distance:>5}  {rate:>7.0f}%  {median:>13}  "
              f"{best if best is not None else '-':>11}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
