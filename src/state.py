from config import SAVE_DIR, EMULATION_SPEED


BEDROOM_STATE_PATH = SAVE_DIR / "bedroom.state"


def load_bedroom_state(pyboy):
    if not BEDROOM_STATE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {BEDROOM_STATE_PATH}. "
            "Create bedroom.state first."
        )

    with open(BEDROOM_STATE_PATH, "rb") as f:
        pyboy.load_state(f)

    pyboy.set_emulation_speed(EMULATION_SPEED)


def save_state(pyboy, path):
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pyboy.save_state(f)


def load_state(pyboy, path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find save state: {path}")

    with open(path, "rb") as f:
        pyboy.load_state(f)

    pyboy.set_emulation_speed(EMULATION_SPEED)