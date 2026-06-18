from emulator import create_emulator
from config import SAVE_DIR


def main():
    save_path = SAVE_DIR / "bedroom.state"

    if not save_path.exists():
        raise FileNotFoundError(
            f"Could not find save state at {save_path}. "
            "Create it first with save_state_test.py."
        )

    pyboy = create_emulator()

    with open(save_path, "rb") as f:
        pyboy.load_state(f)

    print(f"Loaded state from: {save_path}")
    print("The emulator will stay open. Close the window when finished.")

    while pyboy.tick():
        pass

    pyboy.stop()


if __name__ == "__main__":
    main()