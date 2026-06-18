from pyboy import PyBoy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROM_PATH = PROJECT_ROOT / "roms" / "pokemon_red.gb"
SCREENSHOT_PATH = PROJECT_ROOT / "screenshots" / "first_screenshot.png"

def main():
    if not ROM_PATH.exists():
        raise FileNotFoundError(f"Could not find ROM at: {ROM_PATH}")

    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)

    # Run the game for a few seconds worth of frames
    for _ in range(300):
        pyboy.tick()

    image = pyboy.screen.image
    image.save(SCREENSHOT_PATH)

    pyboy.stop()

    print(f"Saved screenshot to: {SCREENSHOT_PATH}")

if __name__ == "__main__":
    main()