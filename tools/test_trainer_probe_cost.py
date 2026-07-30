"""
Measure the cost of repeatedly bumping a known trainer tile, which is the
failure that made workers appear frozen.

    .venv/bin/python3 tools/test_trainer_probe_cost.py

Gen 1 leaves a defeated trainer standing on their tile, still blocking it.
So the winning fight does not open the path, and every later bump in that
direction is a move that cannot succeed. Gating the probe to known trainer
tiles wasn't enough on its own: the probe still fired on each of those
bumps, costing ~11s of emulated time apiece (12 A presses at ~58 ticked
frames each) to rediscover a trainer who is already beaten. A near-random
policy bumps the same tile constantly, so this reads as a freeze -- CPU
pinned, no progress -- rather than as a slowdown.

This walks into a trainer tile many times and reports the per-bump cost.
With the fix, the first bump pays for the probe and the fight, and every
later bump in the same direction is cheap. Without it, every bump costs
roughly the same as the first.
"""

import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from envs.forest_env import PokemonRedForestEnv, KNOWN_TRAINER_TILES
from core.memory import get_player_position
from core.pathfind import walk_to
from actions import get_action_name, num_actions

BUMPS = 8


def direction_index(name):
    for i in range(num_actions()):
        if get_action_name(i) == name:
            return i
    raise ValueError(name)


def main():
    env = PokemonRedForestEnv(max_steps=10_000)
    env.reset()

    target = min(KNOWN_TRAINER_TILES)
    print(f"\nWalking to trainer-adjacent tile {target}")
    reached = walk_to(
        env.pyboy,
        lambda p: (p["x"], p["y"]) == target,
        max_tiles=1500,
    )
    if not reached:
        print("  could not reach it; nothing to measure")
        env.close()
        return 1

    position = get_player_position(env.pyboy)
    print(f"  standing at ({position['x']},{position['y']}) on map {position['map_id']}")

    # walk_to moved the emulator without the env's knowledge, so clear the
    # per-episode probe memory to measure from a clean slate.
    env.probed_trainer_moves = set()

    # Find the direction that is actually blocked -- that's where the
    # trainer is. Try each, and keep the first that doesn't move us.
    blocked_direction = None
    for name in ("up", "down", "left", "right"):
        before = get_player_position(env.pyboy)
        _, _, _, info = env.step(direction_index(name))
        after = get_player_position(env.pyboy)
        if after["map_id"] != before["map_id"]:
            print(f"  stepping {name} left the map; aborting")
            env.close()
            return 1
        if not info["moved"]:
            blocked_direction = name
            break
        # Step back so the next attempt starts from the same tile.
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}[name]
        env.step(direction_index(opposite))

    if blocked_direction is None:
        print("  no blocked direction from this tile; nothing to measure")
        env.close()
        return 1

    print(f"  blocked direction: {blocked_direction}")
    print(f"\nBumping it {BUMPS} times\n")

    action = direction_index(blocked_direction)
    timings = []
    for i in range(BUMPS):
        start = time.time()
        env.step(action)
        elapsed = time.time() - start
        timings.append(elapsed)
        probed = (target, blocked_direction) in env.probed_trainer_moves
        print(f"  bump {i + 1}: {elapsed:6.2f}s   probe already spent: {probed}")

    env.close()

    first = timings[0]
    rest = timings[1:]
    print()
    print(f"  first bump:      {first:.2f}s  (probe + any fight)")
    print(f"  later bumps:     median {statistics.median(rest):.2f}s, "
          f"max {max(rest):.2f}s")
    if statistics.median(rest) < first / 2:
        saved = sum(first - t for t in rest)
        print(f"  later bumps are much cheaper -- the probe is not being repaid")
        print(f"  saved ~{saved:.0f}s across {len(rest)} bumps in this one sequence")
        return 0

    print(f"  later bumps cost about as much as the first -- the probe is")
    print(f"  still firing every time, which is the freeze this guards against")
    return 1


if __name__ == "__main__":
    sys.exit(main())
