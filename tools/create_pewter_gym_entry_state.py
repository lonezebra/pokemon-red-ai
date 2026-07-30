"""
Step through Pewter City's Gym door and save the checkpoint just inside.

    cd src && ../.venv/bin/python3 ../tools/create_pewter_gym_entry_state.py

The city survey found one exit to map 54: tile (16,18), stepping up. Map
54 is Pewter Gym in the Gen 1 map table, but the standing rule (learned
on Route 22, re-earned since) is that a map ID is a hypothesis until a
screenshot confirms it -- so this saves the frame it lands on as
screenshots/pewter_gym_inside.png for eyes-on verification alongside the
checkpoint itself.

Same pattern as create_pewter_city_entry_state's cross(): BFS to the door
tile with walk_to_tile (no trainers on Pewter's streets, so no battle
handling needed out here), one step through, settle, save. The checkpoint
lands just inside the door -- deliberately NOT deeper in, because the Gym
contains a Jr Trainer and Brock, and this state exists so later work can
choose how to approach them rather than having stumbled into either.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state
from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.pathfind import walk_to_tile, _step
from core.memory import get_player_position

DOOR_TILE = (16, 18)
DOOR_DIRECTION = "up"
EXPECTED_MAP = 54
CHECKPOINT = PROJECT_ROOT / "saves" / "pewter_gym_entry.state"
SCREENSHOT = SCREENSHOT_DIR / "pewter_gym_inside.png"


def main():
    pyboy = create_emulator()
    load_state(pyboy, PROJECT_ROOT / "saves" / "pewter_city_entry.state")
    run_frames(pyboy, 30)

    print(f"Walking to the Gym door tile {DOOR_TILE}...")
    if not walk_to_tile(pyboy, *DOOR_TILE):
        print("Could not reach the door tile.")
        return 1

    print(f"Stepping {DOOR_DIRECTION} through the door...")
    _step(pyboy, DOOR_DIRECTION)
    run_frames(pyboy, 30)

    position = get_player_position(pyboy)
    print(f"Landed at: {position}")

    pyboy.screen.image.convert("RGB").save(SCREENSHOT)
    print(f"Saved {SCREENSHOT} -- verify the interior visually before "
          f"trusting the map ID.")

    if position["map_id"] != EXPECTED_MAP:
        print(f"Map is {position['map_id']}, expected {EXPECTED_MAP} -- "
              f"NOT saving a checkpoint.")
        pyboy.stop()
        return 1

    save_state(pyboy, CHECKPOINT)
    print(f"Saved {CHECKPOINT}")
    pyboy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
