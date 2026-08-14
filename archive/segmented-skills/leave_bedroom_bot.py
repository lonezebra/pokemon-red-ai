from core.emulator import create_emulator, run_frames
from core.controls import walk_tile
from core.memory import get_player_position, print_player_position
from core.screen import save_screenshot
from core.state import load_bedroom_state


BEDROOM_MAP_ID = 38


def wait_after_load(pyboy):
    """
    Give the emulator a moment after loading state.
    """
    run_frames(pyboy, 60)


def walk_and_report(pyboy, direction):
    """
    Walk one tile and print the before/after position.
    """

    before = get_player_position(pyboy)
    print(f"Before walking {direction}: {before}")

    walk_tile(pyboy, direction)

    after = get_player_position(pyboy)
    print(f"After walking {direction}:  {after}")

    return before, after


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    wait_after_load(pyboy)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "leave_bedroom_01_start.png")

    if get_player_position(pyboy)["map_id"] != BEDROOM_MAP_ID:
        print("Warning: we are not on the expected bedroom map.")

    print()
    print("Trying route toward bedroom stairs...")

    # Starting around X=3, Y=6.
    # Target is likely around X=7, Y=1.
    route = [
    "right",
    "up",
    "up",
    "up",
    "right",
    ]

    for direction in route:
        walk_and_report(pyboy, direction)
        run_frames(pyboy, 20)

    save_screenshot(pyboy, "leave_bedroom_02_after_route.png")
    print_player_position(pyboy, "Final position")

    print()
    print("Bot finished.")
    print("The emulator will stay open so you can inspect where it ended.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()