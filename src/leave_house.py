from emulator import create_emulator, run_frames
from state import load_bedroom_state
from controls import walk_tile
from memory import get_player_position, print_player_position
from screen import save_screenshot


BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37
PALLET_TOWN_MAP_ID = 0


ROUTE_TO_DOWNSTAIRS = [
    "right",
    "up",
    "up",
    "up",
    "right",
    "up",
    "up",
    "right",
    "right",
]


# From downstairs starting position:
# Map 37, X=7, Y=1
#
# Target exit tile:
# Map 37, X=5, Y=7
#
# So we move:
# left, left, then down 6 times.
ROUTE_DOWNSTAIRS_TO_EXIT = [
    "left",
    "down",
    "down",
    "down",
    "down",
    "down",
    "down",
    "left",
    "left",
    "left",
    "down"
]


def follow_route(pyboy, route, label):
    """
    Follow a route and print position after each step.
    """

    print()
    print(label)
    print("-" * len(label))

    for step_number, direction in enumerate(route, start=1):
        before = get_player_position(pyboy)
        print(f"Step {step_number}: {direction}")
        print(f"  Before: {before}")

        moved = walk_tile(pyboy, direction)
        run_frames(pyboy, 10)

        after = get_player_position(pyboy)
        print(f"  After:  {after}")

        if not moved:
            print(f"  Move failed on step {step_number}: {direction}")
            return False

    return True


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "leave_house_01_start.png")

    print()
    print("Leaving bedroom...")

    reached_downstairs = follow_route(
        pyboy,
        ROUTE_TO_DOWNSTAIRS,
        "Route: bedroom to downstairs",
    )

    if not reached_downstairs:
        print("Failed while going downstairs.")
        pyboy.stop()
        return

    downstairs_pos = get_player_position(pyboy)

    print_player_position(pyboy, "After reaching downstairs")
    save_screenshot(pyboy, "leave_house_02_downstairs.png")

    if downstairs_pos["map_id"] != DOWNSTAIRS_MAP_ID:
        print(f"Expected downstairs map {DOWNSTAIRS_MAP_ID}, got {downstairs_pos['map_id']}.")
        pyboy.stop()
        return

    print()
    print("Walking to house exit...")

    reached_exit = follow_route(
        pyboy,
        ROUTE_DOWNSTAIRS_TO_EXIT,
        "Route: downstairs to house exit",
    )

    if not reached_exit:
        print("Failed while walking to house exit.")
        pyboy.stop()
        return

    print_player_position(pyboy, "At/near house exit")
    save_screenshot(pyboy, "leave_house_03_exit_tile.png")

    # Give the warp time to finish if the final down step triggered the exit.
    run_frames(pyboy, 120)

    final_pos = get_player_position(pyboy)

    print_player_position(pyboy, "Final position")
    save_screenshot(pyboy, "leave_house_04_final.png")

    if final_pos["map_id"] == PALLET_TOWN_MAP_ID:
        print("Success: exited the house and reached Pallet Town.")
    else:
        print("Did not reach Pallet Town.")
        print(f"Expected map {PALLET_TOWN_MAP_ID}, got map {final_pos['map_id']}.")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()