import io

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

    # ram_file=... -- without it, PyBoy reads/writes a
    # `pokemon_red.gb.ram` file next to the ROM by default (its own
    # cartridge battery-save persistence), which every PyBoy instance
    # this project creates would otherwise share since they all point at
    # the same ROM_PATH. Harmless for a single process, but a real race
    # once core/parallel_survey.py started creating several emulators
    # concurrently: one process's write (which truncates the file first)
    # landing between another's open() and read() produced a bare
    # "No data" crash. This project never uses that file anyway -- every
    # bit of state this codebase tracks goes through explicit
    # save_state()/load_state() snapshots of the whole emulator, never
    # PyBoy's own battery-RAM autosave -- so an in-memory, per-instance
    # buffer removes the shared file entirely and changes nothing
    # observable. It can't be empty, though: PyBoy still reads real bytes
    # from whatever it's given at startup (Pokemon Red's cartridge has
    # battery-backed SRAM, so it isn't skipped), it's only that this
    # project's own load_state() call -- always the very next thing that
    # happens -- immediately overwrites it anyway, making the actual
    # content irrelevant. 32768 bytes matches Pokemon Red's real save RAM
    # size (confirmed from the .ram file PyBoy itself had been writing
    # next to the ROM before this fix).
    pyboy = PyBoy(str(ROM_PATH), window=WINDOW_MODE, ram_file=io.BytesIO(bytes(32768)))
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