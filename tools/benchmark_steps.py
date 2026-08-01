"""
Measure how many environment steps per second this machine actually gets.

    # headless
    POKEMON_AI_WINDOW_MODE=null .venv/bin/python3 tools/benchmark_steps.py

    # with a visible window
    POKEMON_AI_WINDOW_MODE=SDL2 .venv/bin/python3 tools/benchmark_steps.py

Run it both ways and compare. The point is to settle whether opening a
window changes throughput, on the hardware in question, rather than
reasoning about it -- and the reasoning is genuinely ambiguous, because a
window can slow things down without showing up as load. If PyBoy's SDL2
renderer presents with vsync, each emulator is capped near the display's
refresh rate and spends the remainder of its time blocked rather than
computing, so CPU and GPU usage both look fine while throughput collapses.
Low utilisation would be the symptom, not evidence against it.

What matters is steps per second, because that is what a training round is
made of. A forest episode is up to 2000 steps, and a round is 400 episodes,
so the arithmetic at the bottom converts the measured rate into a per-round
estimate that can be compared against a round actually taking hours.

Deliberately single-process: one emulator, no workers, no merging. Adding
those in would measure scheduling and contention at the same time, and the
question here is only what one emulator costs.
"""

import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from actions import num_actions
from core.config import WINDOW_MODE, EMULATION_SPEED
from envs.forest_env import PokemonRedForestEnv

STEPS = 300

# What a real forest round costs, for converting a rate into a wall-clock
# estimate. See train_forest_agent.MAX_STEPS and
# train_navigation_parallel.DEFAULT_EPISODES_PER_ROUND.
EPISODE_STEPS = 2000
EPISODES_PER_ROUND = 400


def main():
    print(f"\nwindow mode: {WINDOW_MODE}   emulation speed: {EMULATION_SPEED} "
          f"(0 = unlimited)")

    env = PokemonRedForestEnv(max_steps=STEPS + 1)
    env.reset()
    random.seed(3)

    # A few steps first, so one-off costs (the trainer DQN's first inference,
    # any lazy allocation) don't land inside the timed window.
    for _ in range(10):
        env.step(random.randrange(num_actions()))

    print(f"timing {STEPS} steps...")
    start = time.time()
    for _ in range(STEPS):
        _, _, done, _ = env.step(random.randrange(num_actions()))
        if done:
            env.reset()
    elapsed = time.time() - start
    env.close()

    rate = STEPS / elapsed
    print(f"\n  {STEPS} steps in {elapsed:.1f}s")
    print(f"  {rate:.0f} steps/sec  ({elapsed / STEPS * 1000:.1f} ms/step)")

    worst_case_round = EPISODE_STEPS * EPISODES_PER_ROUND / rate
    print()
    print(f"  one emulator at this rate would take {worst_case_round / 3600:.1f} h")
    print(f"  to run a full {EPISODES_PER_ROUND}-episode round of "
          f"{EPISODE_STEPS}-step episodes;")
    print(f"  divide by the worker count, and note real episodes usually end")
    print(f"  early, so treat this as an upper bound rather than a forecast.")
    print()
    print(f"  Compare this number between window modes. If it barely moves,")
    print(f"  the window is not the bottleneck and the slowness is elsewhere.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
