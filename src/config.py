from pathlib import Path

# This file contains shared settings for the whole project.

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROM_PATH = PROJECT_ROOT / "roms" / "pokemon_red.gb"
SAVE_DIR = PROJECT_ROOT / "saves"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

# PyBoy window options:
# "SDL2" = visible emulator window
# "null" = no visible window, useful later for training faster
WINDOW_MODE = "SDL2"

# Emulation speed:
# 1 = normal speed
# 0 = unlimited speed
EMULATION_SPEED = 0