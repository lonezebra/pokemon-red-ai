from itertools import product

from emulator import create_emulator, run_frames
from config import EMULATION_SPEED
from state import load_bedroom_state
from controls import walk_tile
from memory import get_player_position, print_player_position
from screen import save_screenshot


DOWNSTAIRS_MAP_ID = 37
BEDROOM_MAP_ID = 38

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

DIRECTIONS = [
    "up",
    "down",
    "left",
    "right",
]


def reset_to_downstairs(pyboy):
    """
    Reload bedroom save and walk downstairs.

    Expected result:
        Map 37, X=7, Y=1
    """

    load_bedroom_state(pyboy)

    # Re-apply speed after loading state.
    # Some emulator/window timing behavior can feel reset after load_state().
    pyboy.set_emulation_speed(EMULATION_SPEED)

    run_frames(pyboy, 60)

    for direction in ROUTE_TO_DOWNSTAIRS:
        walk_tile(pyboy, direction)
        run_frames(pyboy, 10)

    pos = get_player_position(pyboy)
    print_player_position(pyboy, "After route to downstairs")

    if pos["map_id"] != DOWNSTAIRS_MAP_ID:
        print("Warning: did not reach expected downstairs map.")

    return pos


def try_sequence(pyboy, sequence):
    """
    Try a candidate sequence from the downstairs starting position.
    """

    print()
    print("=" * 60)
    print(f"Trying house-exit sequence: {sequence}")

    reset_to_downstairs(pyboy)

    starting_map_id = get_player_position(pyboy)["map_id"]

    for direction in sequence:
        before = get_player_position(pyboy)

        print()
        print(f"Move: {direction}")
        print(f"Before: {before}")

        moved = walk_tile(pyboy, direction)
        run_frames(pyboy, 30)

        after = get_player_position(pyboy)
        print(f"After:  {after}")

        if after["map_id"] != starting_map_id:
            print()
            print("Map changed.")

            if after["map_id"] == BEDROOM_MAP_ID:
                print("This went back upstairs to the bedroom, not outside.")
                print("Rejecting this sequence.")
                return False

            print("SUCCESS: map changed to a non-bedroom map. We probably exited the house.")
            save_screenshot(pyboy, f"house_exit_success_{'_'.join(sequence)}.png")
            print(f"Working house-exit sequence: {sequence}")
            print(f"Before final move: {before}")
            print(f"After final move:  {after}")
            return True

        if not moved:
            print("Move was blocked.")

    return False


def main():
    pyboy = create_emulator()

    print("Searching for route from downstairs to outside.")

    for length in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print()
        print("#" * 60)
        print(f"Trying all house-exit sequences of length {length}")

        for sequence_tuple in product(DIRECTIONS, repeat=length):
            sequence = list(sequence_tuple)

            found = try_sequence(pyboy, sequence)

            if found:
                print()
                print("Search found a house-exit route.")
                print("Full route from bedroom:")
                print(ROUTE_TO_DOWNSTAIRS + sequence)

                print()
                print_player_position(pyboy, "Final position")

                print()
                print("The emulator will stay open so you can inspect the result.")

                while pyboy.tick():
                    pass

                pyboy.stop()
                return

    print()
    print("No route found with sequence length up to 6.")
    print_player_position(pyboy, "Final position")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()