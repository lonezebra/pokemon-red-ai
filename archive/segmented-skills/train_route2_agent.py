import os

# Headless unless explicitly asked otherwise, and this must run before any
# core import, because core/config.py reads the variable at import time.
#
# The default is "SDL2", which opens a real Game Boy window. That is what
# you want for watch_*.py, and actively harmful here: it renders every
# frame to the screen at EMULATION_SPEED=0, i.e. as fast as the emulator
# can produce them, dragging a training run that should take minutes into
# a visibly "hung" one, and popping up an unwanted window on a machine
# this might be run on unattended. GIFs are unaffected: the screen buffer
# is still readable with no window.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from envs.route2_env import PokemonRedRoute2Env
from core.config import PROJECT_ROOT
from train_navigation_parallel import train

MODEL_PATH = PROJECT_ROOT / "models" / "route2_q_table.json"
STATE_PATH = PROJECT_ROOT / "models" / "route2_parallel_state.json"

# Route 2 runs from (8, 71) at its southern end up to the Viridian Forest
# south gate at (3, 44) -- 27 tiles of vertical progress, against Route
# 1's 35. Route 1's trained policy solves its route in ~53 steps with an
# 800-step cap, so 600 here is comfortably generous without making failed
# episodes drag.
MAX_STEPS = 600


def main():
    train(
        env_class=PokemonRedRoute2Env,
        model_path=MODEL_PATH,
        state_path=STATE_PATH,
        gif_prefix="route2",
        max_steps=MAX_STEPS,
    )


if __name__ == "__main__":
    main()
