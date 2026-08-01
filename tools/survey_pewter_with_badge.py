"""
Look for Pewter City's Route 3 exit now that the Boulder Badge is in hand.

    cd src && ../.venv/bin/python3 ../tools/survey_pewter_with_badge.py

The badge-less city survey found 696 tiles and 8 exits -- six building
doors and the two Route 2 arrival tiles -- and, conspicuously, nothing to
Route 3, even though Pewter connects east to it. The suspected reason is
Gen 1's script there: an NPC by the east road stops anyone without the
Boulder Badge and marches them back toward the gym, which to a survey
reads as an impassable region rather than an exit. boulder_badge.state is
the first save that can test this.

Steps out of the gym first (boulder_badge.state stands inside, at the
spot Brock was beaten from), saves pewter_city_badged.state outside the
door, then runs the standard survey from it. If the theory is right, the
new meta gains an east-edge exit and more tiles than 696; if the survey
again closes at 696/8, the block is something else and that is worth
knowing precisely.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state
from core.config import PROJECT_ROOT
from core.pathfind import walk_to
from core.memory import get_player_position
from build_map_panorama import build

BADGED_CITY_STATE = PROJECT_ROOT / "saves" / "pewter_city_badged.state"
CITY_MAP = 2
GYM_MAP = 54


def make_badged_city_state():
    pyboy = create_emulator()
    load_state(pyboy, PROJECT_ROOT / "saves" / "boulder_badge.state")
    run_frames(pyboy, 30)

    position = get_player_position(pyboy)
    print(f"Loaded boulder_badge.state at {position}")
    if position["map_id"] == GYM_MAP:
        print("Walking out of the gym...")
        reached = walk_to(
            pyboy,
            lambda p: p["map_id"] == CITY_MAP,
            max_tiles=200,
            stay_on_map=False,
        )
        if not reached:
            print("Could not find the way out of the gym.")
            pyboy.stop()
            return False

    position = get_player_position(pyboy)
    print(f"Standing in the city at {position}")
    save_state(pyboy, BADGED_CITY_STATE)
    print(f"Saved {BADGED_CITY_STATE}")
    pyboy.stop()
    return True


def main():
    if not BADGED_CITY_STATE.exists():
        if not make_badged_city_state():
            return 1
    else:
        print(f"{BADGED_CITY_STATE} already exists, surveying from it.")

    build("pewter_city_badged", "map2_badged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
