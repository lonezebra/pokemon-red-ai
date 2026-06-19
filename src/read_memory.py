from emulator import create_emulator, run_frames
from config import SAVE_DIR


BEDROOM_STATE_PATH = SAVE_DIR / "bedroom.state"


# These are common Pokemon Red / Blue WRAM addresses.
# We will verify them empirically instead of blindly trusting them.
ADDR_PLAYER_Y = 0xD361
ADDR_PLAYER_X = 0xD362
ADDR_MAP_ID = 0xD35E


def load_bedroom_state(pyboy):
    if not BEDROOM_STATE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {BEDROOM_STATE_PATH}. "
            "Create bedroom.state first."
        )

    with open(BEDROOM_STATE_PATH, "rb") as f:
        pyboy.load_state(f)


def print_memory_snapshot(pyboy):
    player_y = pyboy.memory[ADDR_PLAYER_Y]
    player_x = pyboy.memory[ADDR_PLAYER_X]
    map_id = pyboy.memory[ADDR_MAP_ID]

    print()
    print("Memory snapshot")
    print("---------------")
    print(f"Map ID:   {map_id}")
    print(f"Player X: {player_x}")
    print(f"Player Y: {player_y}")


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)

    run_frames(pyboy, 60)

    print_memory_snapshot(pyboy)

    print()
    print("Now move around manually in the emulator window.")
    print("Every second, this script will print the current memory values.")
    print("Close the emulator window when finished.")

    while pyboy.tick():
        # Print about once per second
        if pyboy.frame_count % 60 == 0:
            print_memory_snapshot(pyboy)

    pyboy.stop()


if __name__ == "__main__":
    main() 