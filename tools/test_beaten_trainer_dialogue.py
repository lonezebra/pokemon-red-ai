"""
Check that a failed trainer probe leaves the game controllable.

    .venv/bin/python3 tools/test_beaten_trainer_dialogue.py

Talking to an already-beaten Gen 1 trainer opens their post-battle line
rather than starting a fight, so _try_engage_trainer spends all 12 presses
and returns False. The question this answers is what state it leaves
behind: if the last press opened a text box, the overworld is not
controllable, walk_tile fails in every direction, and the player is stuck
until something presses A again.

That matters because of how the probe is now gated. Restricting it to one
attempt per tile-and-direction per episode was meant to stop paying for a
probe that cannot succeed -- but if a failed probe can leave a text box
open, then refusing to probe again means nothing ever clears it, and the
episode is frozen for its remaining steps. The previous always-probe
behavior would have accidentally recovered by pressing A again on the next
bump. So the gate may have converted a slow loop into a permanent one.

The sequence is deliberately the real thing rather than a stub: walk to a
known trainer tile, fight and win, then probe the same direction again with
the memory cleared, and finally test whether the player can still move.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from actions import get_action_name, num_actions
from core.memory import get_player_position, is_in_battle
from core.pathfind import walk_to, _try_engage_trainer
from core.controls import walk_tile
from envs.forest_env import PokemonRedForestEnv, KNOWN_TRAINER_TILES

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


def can_move_anywhere(pyboy):
    """
    True if some direction actually moves the player, undoing the probe so
    the caller's position is preserved. A text box makes every direction
    fail, which is the signal being tested for.
    """
    opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
    for direction in ("left", "right", "up", "down"):
        before = get_player_position(pyboy)
        if walk_tile(pyboy, direction, verbose=False):
            after = get_player_position(pyboy)
            if after["map_id"] == before["map_id"]:
                walk_tile(pyboy, opposite[direction], verbose=False)
            return True, direction
    return False, None


def main():
    env = PokemonRedForestEnv(max_steps=10_000)
    env.reset()

    target = min(KNOWN_TRAINER_TILES)
    print(f"\nWalking to trainer-adjacent tile {target}")
    if not walk_to(env.pyboy, lambda p: (p["x"], p["y"]) == target, max_tiles=1500):
        print("  could not reach it")
        env.close()
        return 1

    env.probed_trainer_moves = set()
    position = get_player_position(env.pyboy)
    print(f"  at ({position['x']},{position['y']}) on map {position['map_id']}")

    # Find the blocked direction; this also fights the trainer.
    blocked = None
    for name in ("up", "down", "left", "right"):
        before = get_player_position(env.pyboy)
        _, _, _, info = env.step(direction_index(name))
        after = get_player_position(env.pyboy)
        if after["map_id"] != before["map_id"]:
            print(f"  stepping {name} left the map; aborting")
            env.close()
            return 1
        if not info["moved"]:
            blocked = name
            break
        env.step(direction_index({"up": "down", "down": "up",
                                  "left": "right", "right": "left"}[name]))

    if blocked is None:
        print("  no blocked direction here")
        env.close()
        return 1

    print(f"  blocked direction: {blocked} (trainer fought during discovery)")

    print("\nControl state after the fight\n")
    movable, via = can_move_anywhere(env.pyboy)
    check(
        "player can move after the trainer battle",
        movable,
        fail_detail="every direction blocked -- already stuck before re-probing",
        ok_detail=f"moved {via}",
    )

    # Now the trainer is beaten. Probe again, exactly as the old
    # always-probe behavior would on the next bump, and see what it leaves.
    print("\nRe-probing a beaten trainer (what the old code did every bump)\n")
    walk_tile(env.pyboy, blocked, verbose=False)  # bump into them
    started_battle = _try_engage_trainer(env.pyboy)
    check(
        "no battle starts against a beaten trainer",
        not started_battle,
        fail_detail="a battle started, so this trainer was not actually beaten",
        ok_detail="probe exhausted its presses without a battle, as expected",
    )
    check(
        "not left inside a battle",
        not is_in_battle(env.pyboy),
        fail_detail="still in battle after the probe",
        ok_detail="not in battle",
    )

    movable, via = can_move_anywhere(env.pyboy)
    check(
        "player can still move after a failed probe",
        movable,
        fail_detail="EVERY direction blocked -- a text box is open and nothing "
                    "will clear it, so gating the probe freezes the episode",
        ok_detail=f"moved {via} -- the failed probe leaves the overworld usable, "
                  f"so gating it is safe",
    )

    env.close()

    print()
    if failures:
        print(f"{len(failures)} failure(s): {failures}")
        print("If the last check failed, _step must guarantee control is back")
        print("after a failed probe rather than assuming it.")
        return 1
    print("A failed probe leaves the overworld controllable, so probing once")
    print("per tile-and-direction cannot strand the agent in dialogue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
