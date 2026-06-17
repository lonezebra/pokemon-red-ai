from pyboy import PyBoy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = PROJECT_ROOT / "roms" / "pokemon_red.gb"

def main():
    if not ROM_PATH.exists():
        raise FileNotFoundError(f"Could not find ROM at: {ROM_PATH}")

    pyboy = PyBoy(str(ROM_PATH), window="SDL2")
    pyboy.set_emulation_speed(1)

    print("Pokemon Red is running.")
    print("Close the emulator window to stop.")

    while pyboy.tick():
        pass

    pyboy.stop()

if __name__ == "__main__":
    main()