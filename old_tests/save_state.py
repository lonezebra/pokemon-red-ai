from emulator import create_emulator
from config import SAVE_DIR


def main():
    SAVE_DIR.mkdir(exist_ok=True)

    save_path = SAVE_DIR / "bedroom.state"

    pyboy = create_emulator()

    print("Manual timed save-state helper.")
    print()
    print("The emulator should open now.")
    print("Click inside the emulator window and play manually.")
    print()
    print("This script will automatically save after about 5 minutes.")
    print("Use that time to start a new game and get to the bedroom.")
    print()
    print("After saving, the emulator will close.")
    print()

    # Pokemon Red runs at roughly 60 frames per second.
    # 60 frames * 60 seconds * 5 minutes = 18,000 frames.
    frames_until_save = 60 * 60 * 3

    try:
        for frame in range(frames_until_save):
            pyboy.tick()

            # Print progress once per minute.
            if frame > 0 and frame % (60 * 60) == 0:
                minutes_elapsed = frame // (60 * 60)
                minutes_left = 5 - minutes_elapsed
                print(f"{minutes_elapsed} minute(s) elapsed. About {minutes_left} minute(s) left.")

        print()
        print("Timer finished. Saving state...")

        with open(save_path, "wb") as f:
            pyboy.save_state(f)

        print(f"Saved state to: {save_path}")

    finally:
        pyboy.stop()
        print("Emulator stopped.")


if __name__ == "__main__":
    main()