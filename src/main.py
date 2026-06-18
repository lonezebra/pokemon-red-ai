from emulator import create_emulator, run_frames
from controls import press_button, press_sequence
from screen import save_screenshot


def main():
    pyboy = create_emulator()

    print("Pokemon Red AI project started.")
    print("Waiting for the game to boot...")

    # Wait about 10 seconds.
    run_frames(pyboy, 720)

    save_screenshot(pyboy, "after_boot.png")

    print("Trying a slow button sequence...")

    # This is intentionally slow so you can see what is happening.
    press_sequence(
        pyboy,
        ["start", "a", "a", "start"],
        hold_frames=30,
        release_frames=90,
    )

    save_screenshot(pyboy, "after_button_sequence.png")

    print("Now trying a few movement inputs...")

    press_button(pyboy, "up", hold_frames=30, release_frames=30)
    press_button(pyboy, "right", hold_frames=30, release_frames=30)
    press_button(pyboy, "down", hold_frames=30, release_frames=30)
    press_button(pyboy, "left", hold_frames=30, release_frames=30)

    save_screenshot(pyboy, "after_movement_test.png")

    print("Test complete.")
    print("The emulator will stay open. Close the window when finished.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()