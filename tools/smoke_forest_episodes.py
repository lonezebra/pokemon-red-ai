"""
Run real Viridian Forest episodes and check none of them go dead partway.

    .venv/bin/python3 tools/smoke_forest_episodes.py

The unit tests cover each fix in isolation against stubs or a single
contrived situation. This is the integration check: actual episodes, actual
emulator, actual trainers, looking for the signature of the bug that made
workers appear stuck.

That signature is specific. A frozen episode does not crash -- it runs its
full step budget while the player stays on one tile, because an open text
box makes every direction read as blocked. So the tell is an episode that
used all its steps while visiting almost no tiles, and the check is that
tiles visited stays proportionate to steps taken.

The agent is walked to a trainer first rather than left to find one. A
purely random start does not work here, and finding that out was itself
worth knowing: the forest entry tile (17,47) sits on the map-50 exit row,
so a random walk steps out of the forest and ends the episode within a
handful of steps roughly four times in five. Six random episodes reached
zero trainers. Walking to a known trainer tile, fighting, and only then
walking randomly is what actually puts the fix under test.
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from actions import get_action_name, num_actions
from core.memory import get_player_position
from core.pathfind import walk_to
from envs.forest_env import PokemonRedForestEnv, KNOWN_TRAINER_TILES

STEPS_AFTER_FIGHT = 120

# A stranded agent visits essentially one tile however long it runs, because
# an open text box makes every direction read as blocked. Real movement over
# this many steps covers far more ground than this.
MIN_TILES_AFTER_FIGHT = 8

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

failures = []


def check(label, ok, fail_detail="", ok_detail=""):
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def direction_index(name):
    for i in range(num_actions()):
        if get_action_name(i) == name:
            return i
    raise ValueError(name)


def main():
    env = PokemonRedForestEnv(max_steps=100_000)
    env.reset()
    random.seed(11)

    target = min(KNOWN_TRAINER_TILES)
    print(f"\nWalking to trainer tile {target}")
    if not walk_to(env.pyboy, lambda p: (p["x"], p["y"]) == target, max_tiles=1500):
        print("  could not reach it")
        env.close()
        return 1

    env.probed_trainer_moves = set()
    env.visited_positions = set()

    # Engage the trainer by finding the blocked direction from this tile.
    blocked = None
    for name in ("up", "down", "left", "right"):
        before = get_player_position(env.pyboy)
        _, _, _, info = env.step(direction_index(name))
        if get_player_position(env.pyboy)["map_id"] != before["map_id"]:
            print(f"  stepping {name} left the map; aborting")
            env.close()
            return 1
        if not info["moved"]:
            blocked = name
            break
        env.step(direction_index(OPPOSITE[name]))

    check(
        "engaged a trainer",
        blocked is not None and bool(env.probed_trainer_moves),
        fail_detail="no trainer engaged, so the fix was never exercised",
        ok_detail=f"fought the trainer blocking {blocked}",
    )
    if blocked is None:
        env.close()
        return 1

    print(f"\nBumping the beaten trainer again, then walking randomly\n")

    # This is the exact sequence that used to strand the agent: bump the
    # now-beaten trainer, whose post-battle line opens a text box.
    env.step(direction_index(blocked))

    tiles_before = len(env.visited_positions)
    for _ in range(STEPS_AFTER_FIGHT):
        _, _, done, _ = env.step(random.randrange(num_actions()))
        if done:
            break

    position = get_player_position(env.pyboy)
    tiles_after = len(env.visited_positions)
    gained = tiles_after - tiles_before

    print(f"  tiles visited after the re-bump: {gained}")
    print(f"  ended at ({position['x']},{position['y']}) on map {position['map_id']}")

    check(
        "keeps moving after re-bumping a beaten trainer",
        gained >= MIN_TILES_AFTER_FIGHT,
        fail_detail=f"only {gained} new tiles in {STEPS_AFTER_FIGHT} steps -- "
                    f"stranded, which is the freeze this guards against",
        ok_detail=f"{gained} new tiles -- control was not lost to the dialogue",
    )

    env.close()

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        return 1
    print("A trainer encounter no longer strands the agent: after fighting and")
    print("re-bumping the beaten trainer, the episode keeps exploring normally.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
