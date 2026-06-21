from emulator import create_emulator, run_frames
from state import load_bedroom_state
from controls import walk_tile
from memory import get_player_position, print_player_position
from screen import save_screenshot


BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37


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


def follow_route(pyboy, route):
    """
    Follow a list of one-tile movement directions.

    Returns:
        True if every move succeeded.
        False if any move failed.
    """

    for step_number, direction in enumerate(route, start=1):
        print()
        print(f"Route step {step_number}/{len(route)}: {direction}")

        before = get_player_position(pyboy)
        print(f"Before: {before}")

        moved = walk_tile(pyboy, direction)
        run_frames(pyboy, 10)

        after = get_player_position(pyboy)
        print(f"After:  {after}")

        if not moved:
            print(f"Move failed on step {step_number}: {direction}")
            return False

    return True


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "leave_bedroom_clean_01_start.png")

    start_pos = get_player_position(pyboy)

    if start_pos["map_id"] != BEDROOM_MAP_ID:
        print(f"Warning: expected bedroom map {BEDROOM_MAP_ID}, got {start_pos['map_id']}.")

    print()
    print("Following confirmed route to downstairs...")

    route_success = follow_route(pyboy, ROUTE_TO_DOWNSTAIRS)

    final_pos = get_player_position(pyboy)

    print()
    print_player_position(pyboy, "Final position")
    save_screenshot(pyboy, "leave_bedroom_clean_02_final.png")

    if route_success and final_pos["map_id"] == DOWNSTAIRS_MAP_ID:
        print("Success: reached downstairs.")
    elif final_pos["map_id"] == DOWNSTAIRS_MAP_ID:
        print("Reached downstairs, but one move reported failure.")
    else:
        print("Did not reach downstairs.")
        print(f"Expected map {DOWNSTAIRS_MAP_ID}, got map {final_pos['map_id']}.")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()