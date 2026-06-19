from emulator import create_emulator, run_frames
from state import load_bedroom_state
from memory import get_player_position
from controls import press_button


def test_hold_frames(pyboy, direction, hold_frames):
    """
    Load the bedroom state, press one direction with a specific hold duration,
    then report how far the player moved.
    """

    load_bedroom_state(pyboy)
    run_frames(pyboy, 60)

    before = get_player_position(pyboy)

    press_button(
        pyboy,
        direction,
        hold_frames=hold_frames,
        release_frames=30,
    )

    run_frames(pyboy, 30)

    after = get_player_position(pyboy)

    dx = after["x"] - before["x"]
    dy = after["y"] - before["y"]

    print()
    print(f"Hold frames: {hold_frames}")
    print(f"Before: {before}")
    print(f"After:  {after}")
    print(f"Delta:  dx={dx}, dy={dy}")

    return before, after


def main():
    pyboy = create_emulator()

    print("Movement calibration test")
    print("-------------------------")
    print("Testing how many frames are needed to move exactly one tile.")
    print()

    direction = "right"

    for hold_frames in [18, 20, 22, 24, 26, 28, 30, 32]:
        test_hold_frames(pyboy, direction, hold_frames)

    print()
    print("Calibration finished.")
    print("Look for the smallest hold_frames value where:")
    print("  X increases by exactly 1")
    print()
    print("The emulator will stay open so you can inspect the final state.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()