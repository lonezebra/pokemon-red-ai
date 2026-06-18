from pyboy import PyBoy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = PROJECT_ROOT / "roms" / "pokemon_red.gb"


def run_frames(pyboy, frames):
    """
    Advance the emulator by a certain number of frames.

    Pokemon Red runs at roughly 60 frames per second.
    So:
      60 frames  = about 1 second
      300 frames = about 5 seconds
      600 frames = about 10 seconds
    """
    for _ in range(frames):
        pyboy.tick()


def press_button(pyboy, button, hold_frames=20, release_frames=20):
    """
    Press and release a Game Boy button.

    hold_frames controls how long the button is held down.
    release_frames gives the game time to notice that the button was released.
    """
    print(f"Pressing {button}")

    pyboy.button_press(button)
    run_frames(pyboy, hold_frames)

    pyboy.button_release(button)
    run_frames(pyboy, release_frames)


def main():
    if not ROM_PATH.exists():
        raise FileNotFoundError(f"Could not find ROM at: {ROM_PATH}")

    pyboy = PyBoy(str(ROM_PATH), window="SDL2")
    pyboy.set_emulation_speed(1)

    print("Game started.")
    print("Waiting longer for Pokemon Red to reach the title/menu screens...")

    # Wait about 10 seconds.
    # The original script only waited about 2 seconds, which was too short.
    run_frames(pyboy, 600)

    print("Trying to move through title/menu screens...")

    # These are intentionally slow and repeated.
    # We are not trying to be elegant yet; we are testing that input works.
    press_button(pyboy, "start", hold_frames=30)
    run_frames(pyboy, 120)

    press_button(pyboy, "a", hold_frames=30)
    run_frames(pyboy, 120)

    press_button(pyboy, "a", hold_frames=30)
    run_frames(pyboy, 120)

    press_button(pyboy, "start", hold_frames=30)
    run_frames(pyboy, 120)

    print("Button test finished.")
    print("The emulator will stay open. The bot is no longer pressing buttons.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()