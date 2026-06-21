from emulator import create_emulator
from state import load_bedroom_state
from memory import print_player_position


def main():
    pyboy = create_emulator()

    print("Loading bedroom state...")
    load_bedroom_state(pyboy)

    print("Manual position logger.")
    print("Click the emulator window and move manually.")
    print("Your position will print once per second.")
    print("Close the emulator window when finished.")

    while pyboy.tick():
        if pyboy.frame_count % 60 == 0:
            print_player_position(pyboy, "Current position")

    pyboy.stop()


if __name__ == "__main__":
    main()