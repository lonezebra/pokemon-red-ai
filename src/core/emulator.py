from pyboy import PyBoy

from core.config import ROM_PATH, WINDOW_MODE, EMULATION_SPEED

#loading save states
#resetting the game
#running the emulator faster
#closing the emulator cleanly


def create_emulator():
    """
    Create and configure a PyBoy emulator instance.

    This function is responsible for:
    1. Checking that the ROM exists.
    2. Starting PyBoy.
    3. Setting the emulation speed.
    4. Returning the emulator object so other files can use it.
    """

    if not ROM_PATH.exists():
        raise FileNotFoundError(
            f"Could not find ROM at: {ROM_PATH}\n"
            "Make sure your ROM is named pokemon_red.gb and is inside the roms/ folder."
        )

    pyboy = PyBoy(str(ROM_PATH), window=WINDOW_MODE)
    pyboy.set_emulation_speed(EMULATION_SPEED)

    return pyboy


def run_frames(pyboy, frames):
    """
    Advance the emulator by a specific number of frames.

    Pokemon Red runs at roughly 60 frames per second.

    Examples:
        60 frames  = about 1 second
        300 frames = about 5 seconds
        600 frames = about 10 seconds
    """

    for _ in range(frames):
        pyboy.tick()