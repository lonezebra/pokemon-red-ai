from emulator import create_emulator, run_frames
from state import load_bedroom_state
from navigation import walk_to
from controls import walk_until_position_changes
from memory import print_player_position, get_player_position
from screen import save_screenshot


BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "nav_leave_bedroom_01_start.png")

    # Walk to the tile near the stairs.
    reached = walk_to(pyboy, target_x=4, target_y=3)

    if not reached:
        print("Could not reach the stair approach tile.")
        while pyboy.tick():
            pass
        pyboy.stop()
        return

    print()
    print("Now stepping onto the stairs...")
    walk_until_position_changes(pyboy, "right")
    run_frames(pyboy, 120)

    print_player_position(pyboy, "After stepping onto stairs")
    save_screenshot(pyboy, "nav_leave_bedroom_02_after_stairs.png")

    final_pos = get_player_position(pyboy)

    if final_pos["map_id"] == DOWNSTAIRS_MAP_ID:
        print("Success: reached downstairs map.")
    else:
        print("Not downstairs yet.")
        print(f"Expected map {DOWNSTAIRS_MAP_ID}, got map {final_pos['map_id']}.")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()