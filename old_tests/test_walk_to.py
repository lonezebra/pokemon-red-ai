from emulator import create_emulator, run_frames
from state import load_bedroom_state
from memory import print_player_position
from navigation import walk_to
from screen import save_screenshot


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    print_player_position(pyboy, "Starting position")
    save_screenshot(pyboy, "walk_to_01_start.png")

    # From your tests:
    # Bedroom start is X=3, Y=6.
    # The useful open tile before the stairs path is X=4, Y=3.
    success = walk_to(pyboy, target_x=4, target_y=3)

    print()
    print(f"walk_to success: {success}")
    print_player_position(pyboy, "After walk_to")
    save_screenshot(pyboy, "walk_to_02_after.png")

    print()
    print("The emulator will stay open so you can inspect the result.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()