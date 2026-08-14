from core.emulator import create_emulator, run_frames
from core.config import PROJECT_ROOT
from core.state import load_state, save_state
from core.pathfind import walk_to_map
from core.memory import get_player_position, print_player_position
from core.screen import save_screenshot
from create_starter_obtained_state import wait_for_control_and_walk
from create_pokedex_obtained_state import POKEDEX_OBTAINED_STATE_PATH

# Reaches the real Route 2 -- map 13, north of Viridian City -- and saves
# it as the starting checkpoint for the Route 2 navigation milestone.
#
# An earlier script with this name reached map 33 instead and called it
# Route 2. That was Route 22, west of Viridian and a dead end until all
# eight badges; a whole 1500-episode training run was spent on it before
# the mistake was caught. See create_route22_entry_state.py, which keeps
# that route (it matters later, for Victory Road) and records how the
# misidentification happened.
#
# The reason the real Route 2 was unreachable at all is that Viridian's
# north exit stays shut until Oak's Parcel is delivered, so this starts
# from saves/pokedex_obtained.state -- see
# create_pokedex_obtained_state.py, which both runs that errand and
# measures the gate opening.
#
# Route 2 is entered at its *southern* end, around (7-9, 71), and runs
# north toward Viridian Forest. That y=71 is worth noting for whoever
# builds the training task: like Route 1, this is a tall vertical
# corridor, so the y-coordinate potential shaping that fixed Route 1
# (see rewards/route1_rewards.py) should carry over directly -- but this
# time confirm which way is forward before building on it.

ROUTE_2_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "route2_entry.state"
ROUTE_2_MAP_ID = 13

PALLET_TOWN_MAP_ID = 0
ROUTE_1_MAP_ID = 12
VIRIDIAN_CITY_MAP_ID = 1


def main():
    pyboy = create_emulator()
    load_state(pyboy, POKEDEX_OBTAINED_STATE_PATH)
    run_frames(pyboy, 30)

    # pokedex_obtained.state is saved inside Oak's Lab; make sure the
    # scene has really handed control back before pathfinding.
    wait_for_control_and_walk(pyboy, "down")
    print_player_position(pyboy, "Starting from")

    for map_id, label in (
        (PALLET_TOWN_MAP_ID, "Pallet Town"),
        (ROUTE_1_MAP_ID, "Route 1"),
        (VIRIDIAN_CITY_MAP_ID, "Viridian City"),
        (ROUTE_2_MAP_ID, "Route 2"),
    ):
        if not walk_to_map(pyboy, map_id):
            print(f"Warning: could not reach {label} -- aborting.")
            pyboy.stop()
            return
        print(f"  reached {label}: {get_player_position(pyboy)}")

    print_player_position(pyboy, "Arrived on Route 2")
    save_screenshot(pyboy, "route2_entry.png")
    save_state(pyboy, ROUTE_2_ENTRY_STATE_PATH)
    print(f"Saved {ROUTE_2_ENTRY_STATE_PATH}")

    pyboy.stop()


if __name__ == "__main__":
    main()
