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

    # Finding the direction already engaged and beat the trainer, so from
    # here the move is permanently blocked and any probe is guaranteed to
    # burn all 12 presses without finding a battle. That is exactly the
    # worst case worth measuring.
    action = direction_index(blocked_direction)

    def bump(clear_memory):
        if clear_memory:
            # Reproduces the old behavior: nothing remembers that this
            # probe was already paid for, so _step pays it again.
            env.probed_trainer_moves.discard((target, blocked_direction))
        start = time.time()
        env.step(action)
        return time.time() - start

    print(f"\nOld behavior -- probe memory cleared before each bump\n")
    without_memory = []
    for i in range(BUMPS):
        elapsed = bump(clear_memory=True)
        without_memory.append(elapsed)
        print(f"  bump {i + 1}: {elapsed:6.2f}s")

    print(f"\nNew behavior -- probe paid once per tile+direction per episode\n")
    with_memory = []
    for i in range(BUMPS):
        elapsed = bump(clear_memory=False)
        with_memory.append(elapsed)
        print(f"  bump {i + 1}: {elapsed:6.2f}s")

    env.close()

    old_median = statistics.median(without_memory)
    new_median = statistics.median(with_memory)
    print()
    print(f"  re-probing every bump: median {old_median:6.2f}s")
    print(f"  probing once:          median {new_median:6.2f}s")

    if old_median <= 0:
        print("  could not measure")
        return 1

    print(f"  speedup on a blocked bump: {old_median / max(new_median, 1e-6):.0f}x")

    # A 2000-step episode spent bumping a beaten trainer, which is what a
    # near-random policy does, is the case that looked like a freeze.
    print()
    print(f"  extrapolated over 2000 steps of bumping this one tile:")
    print(f"    re-probing: {old_median * 2000 / 60:7.1f} min")
    print(f"    probing once: {new_median * 2000 / 60:5.1f} min")

    if new_median < old_median / 5:
        print()
        print("  Confirmed: the repeated probe was the cost, and it is gone.")
        return 0

    print()
    print("  Not the dominant cost here -- look elsewhere for the freeze.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
