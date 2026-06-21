from itertools import product

from emulator import create_emulator, run_frames
from state import load_bedroom_state
from controls import walk_tile
from memory import get_player_position, print_player_position
from screen import save_screenshot


BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37

# This is the precise one-tile version of your known route.
# It gets us to X=5, Y=3.
APPROACH_ROUTE = [
    "right",
    "up",
    "up",
    "up",
    "right",
]


DIRECTIONS = [
    "up",
    "down",
    "left",
    "right",
]


def reset_to_stair_area(pyboy):
    """
    Reload bedroom save and walk to the known stair-area position.

    Expected result:
        Map 38, X=5, Y=3
    """

    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    for direction in APPROACH_ROUTE:
        walk_tile(pyboy, direction)
        run_frames(pyboy, 120)

    pos = get_player_position(pyboy)
    print_player_position(pyboy, "After approach route")

    return pos


def try_sequence(pyboy, sequence):
    """
    Try a candidate sequence from the stair-area position.
    """

    print()
    print("=" * 60)
    print(f"Trying continuation sequence: {sequence}")

    reset_to_stair_area(pyboy)

    for direction in sequence:
        before = get_player_position(pyboy)

        print()
        print(f"Move: {direction}")
        print(f"Before: {before}")

        moved = walk_tile(pyboy, direction)
        run_frames(pyboy, 240)

        after = get_player_position(pyboy)
        print(f"After:  {after}")

        if after["map_id"] == DOWNSTAIRS_MAP_ID:
            print()
            print("SUCCESS: reached downstairs.")
            save_screenshot(pyboy, f"stairs_success_{'_'.join(sequence)}.png")
            print(f"Working continuation sequence: {sequence}")
            return True

        if after["map_id"] != BEDROOM_MAP_ID:
            print()
            print(f"Map changed to unexpected map: {after['map_id']}")
            save_screenshot(pyboy, f"stairs_other_map_{'_'.join(sequence)}.png")
            print(f"Map-changing continuation sequence: {sequence}")
            return True

        if not moved:
            print("Move was blocked.")

    return False


def main():
    pyboy = create_emulator()

    print("Searching for bedroom stair transition from known stair area.")
    print("Known approach route:")
    print(APPROACH_ROUTE)

    # Try sequences of length 1, then 2, then 3.
    # This keeps the search small and easy to read.
    for length in [1, 2, 3, 4, 5]:
        print()
        print("#" * 60)
        print(f"Trying all continuation sequences of length {length}")

        for sequence_tuple in product(DIRECTIONS, repeat=length):
            sequence = list(sequence_tuple)

            found = try_sequence(pyboy, sequence)

            if found:
                print()
                print("Search found a route.")
                print(f"Full route:")
                print(APPROACH_ROUTE + sequence)

                print()
                print_player_position(pyboy, "Final position")

                print()
                print("The emulator will stay open so you can inspect the result.")

                while pyboy.tick():
                    pass

                pyboy.stop()
                return

    print()
    print("No route found with continuation length up to 3.")
    print_player_position(pyboy, "Final position")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()