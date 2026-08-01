"""
Find which input actually clears a beaten trainer's dialogue box.

    .venv/bin/python3 tools/find_dialogue_recovery.py

test_beaten_trainer_dialogue.py established that a failed probe against an
already-beaten trainer leaves a text box open, with every direction blocked.
_step therefore has to restore control itself rather than assume it. What it
should press is not obvious, and guessing would be a poor way to choose:

  - A advances Gen 1 text, but it is also what starts a conversation, so
    pressing it while still facing the trainer can close one text box and
    immediately open the next.
  - B also advances text but does not initiate dialogue, so it may settle
    where A oscillates.
  - wait_for_free_movement already exists for "a text box nothing else
    clears", but it recovers by walking, and it deliberately does not walk
    back when the successful step crossed a map boundary -- from a tile near
    an exit that would silently move the agent to another map, which in the
    forest env means ending the episode.

Rather than reason about it, this reaches the stuck state once, saves it,
and then restores that exact state to try each candidate independently. The
save matters: reaching the state requires a BFS walk of several minutes, and
every strategy has to be judged from the identical starting point to be
comparable at all.
"""

import io
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from actions import get_action_name, num_actions
from core.controls import press_button, walk_tile, wait_for_free_movement
from core.emulator import run_frames
from core.memory import get_player_position
from core.pathfind import walk_to, _try_engage_trainer
from envs.forest_env import PokemonRedForestEnv, KNOWN_TRAINER_TILES

OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}


def direction_index(name):
    for i in range(num_actions()):
        if get_action_name(i) == name:
            return i
    raise ValueError(name)


def snapshot(pyboy):
    buf = io.BytesIO()
    pyboy.save_state(buf)
    return buf.getvalue()


def restore(pyboy, data):
    buf = io.BytesIO(data)
    buf.seek(0)
    pyboy.load_state(buf)
    run_frames(pyboy, 2)


def movement_restored(pyboy):
    """
    Whether any direction moves the player, restoring position afterwards.
    Reports the map too, so a strategy that "works" by walking off the map
    is not mistaken for a success.
    """
    start_map = get_player_position(pyboy)["map_id"]
    for direction in ("left", "right", "up", "down"):
        before = get_player_position(pyboy)
        if walk_tile(pyboy, direction, verbose=False):
            after = get_player_position(pyboy)
            if after["map_id"] == before["map_id"]:
                walk_tile(pyboy, OPPOSITE[direction], verbose=False)
                return True, start_map == get_player_position(pyboy)["map_id"]
            return True, False
    return False, True


def press_b(pyboy, times=8):
    for _ in range(times):
        press_button(pyboy, "b", hold_frames=12, release_frames=24)


def press_a(pyboy, times=8):
    for _ in range(times):
        press_button(pyboy, "a", hold_frames=12, release_frames=24)


def press_b_then_a(pyboy):
    press_b(pyboy, times=4)
    press_a(pyboy, times=4)


STRATEGIES = [
    ("press B x8", press_b),
    ("press A x8", press_a),
    ("press B x4 then A x4", press_b_then_a),
    ("wait_for_free_movement", lambda p: wait_for_free_movement(p)),
    ("nothing (control)", lambda p: None),
]


def main():
    env = PokemonRedForestEnv(max_steps=10_000)
    env.reset()

    target = min(KNOWN_TRAINER_TILES)
    print(f"\nReaching the stuck state at trainer tile {target}")
    if not walk_to(env.pyboy, lambda p: (p["x"], p["y"]) == target, max_tiles=1500):
        print("  could not reach it")
        env.close()
        return 1

    env.probed_trainer_moves = set()

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

    if blocked is None:
        print("  no blocked direction here")
        env.close()
        return 1

    # Re-probe the now-beaten trainer to open the dialogue, then freeze that
    # exact moment so every strategy starts from it.
    walk_tile(env.pyboy, blocked, verbose=False)
    _try_engage_trainer(env.pyboy)
    stuck = snapshot(env.pyboy)

    movable, _ = movement_restored(env.pyboy)
    print(f"  stuck state captured (movable={movable}, expected False)")
    if movable:
        print("  did not reproduce the stuck state; nothing to compare")
        env.close()
        return 1

    print(f"\nTrying each recovery from the identical stuck state\n")
    results = []
    for label, strategy in STRATEGIES:
        restore(env.pyboy, stuck)
        strategy(env.pyboy)
        movable, same_map = movement_restored(env.pyboy)
        position = get_player_position(env.pyboy)
        ok = movable and same_map
        status = "RECOVERED" if ok else ("MOVED OFF MAP" if movable else "still stuck")
        print(f"  {label:26s} {status:14s} at ({position['x']},{position['y']}) "
              f"map {position['map_id']}")
        results.append((label, ok))

    env.close()

    winners = [label for label, ok in results if ok and label != "nothing (control)"]
    control_recovered = any(ok for label, ok in results if label == "nothing (control)")

    print()
    if control_recovered:
        print("The control case recovered on its own, so this run did not actually")
        print("reproduce a stuck state and the comparison proves nothing.")
        return 1
    if not winners:
        print("Nothing tried restored control on the same map. _step needs a")
        print("different approach than any of these.")
        return 1

    print(f"Use: {winners[0]}")
    if len(winners) > 1:
        print(f"(also worked: {', '.join(winners[1:])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
