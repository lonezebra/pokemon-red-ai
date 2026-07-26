import os
from pathlib import Path

# This file contains shared settings for the whole project.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ROM_PATH = PROJECT_ROOT / "roms" / "pokemon_red.gb"
SAVE_DIR = PROJECT_ROOT / "saves"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

# PyBoy window options:
# "SDL2" = visible emulator window
# "null" = no visible window, needed for headless/fast training runs
#
# Override without editing this file, e.g.:
#   POKEMON_AI_WINDOW_MODE=null python src/train_battle_agent.py
WINDOW_MODE = os.environ.get("POKEMON_AI_WINDOW_MODE", "SDL2")

# Emulation speed:
# 1 = normal speed
# 0 = unlimited speed
EMULATION_SPEED = 0