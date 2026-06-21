from emulator import create_emulator, run_frames
from state import load_bedroom_state
from navigation import walk_to
from controls import walk_tile
from memory import get_player_position, print_player_position
from screen import save_screenshot


BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37


def reload_and_go_to_approach(pyboy):
    """
    Reload the bedroom state and walk to the known nearby area.
    """
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    reached = walk_to(pyboy, target_x=4, target_y=3)

    if not reached:
        print("Could not reach X=4,Y=3.")
        return False

    return True


def try_sequence(pyboy, sequence):
    """
    Try a sequence of moves from the bedroom save.

    Returns True if the sequence reaches the downstairs map.
    """
    print()
    print("=" * 50)
    print(f"Trying sequence: {sequence}")

    if not reload_and_go_to_approach(pyboy):
        return False

    print_player_position(pyboy, "Starting test position")

    for direction in sequence:
        print()
        print(f"Move: {direction}")

        before = get_player_position(pyboy)
        print(f"Before: {before}")

        moved = walk_tile(pyboy, direction)
        run_frames(pyboy, 240)

        after = get_player_position(pyboy)
        print(f"After:  {after}")

        if not moved:
            print("Move did not change position.")

        if after["map_id"] == DOWNSTAIRS_MAP_ID:
            print("SUCCESS: reached downstairs.")
            save_screenshot(pyboy, f"stairs_success_{'_'.join(sequence)}.png")
            return True

        if after["map_id"] != BEDROOM_MAP_ID:
            print(f"Map changed, but not to expected downstairs map: {after['map_id']}")
            save_screenshot(pyboy, f"stairs_other_map_{'_'.join(sequence)}.png")
            return True

    print("Sequence did not find stairs.")
    return False


def main():
    pyboy = create_emulator()

    candidate_sequences = [
        ["right"],
        ["right", "right"],
        ["right", "right", "right"],
        ["right", "right", "right", "right"],

        ["right", "right", "up"],
        ["right", "right", "right", "up"],
        ["right", "right", "right", "right", "up"],

        ["right", "right", "down"],
        ["right", "right", "right", "down"],
        ["right", "right", "right", "right", "down"],

        ["right", "right", "right", "left"],
        ["right", "right", "right", "right", "left"],
    ]

    for sequence in candidate_sequences:
        found = try_sequence(pyboy, sequence)

        if found:
            print()
            print(f"Found working sequence: {sequence}")
            break

    print()
    print("Search finished.")
    print_player_position(pyboy, "Final position")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()