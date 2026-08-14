import os

# Headless unless explicitly asked otherwise, and this must run before any
# core import, because core/config.py reads the variable at import time.
#
# The default is "SDL2", which opens a real Game Boy window per emulator.
# That is what you want for watch_*.py, and actively harmful here: training
# creates one emulator per worker, so on a machine with many cores it opens
# that many windows and renders every frame of each to the screen at
# EMULATION_SPEED=0, i.e. as fast as the emulator can produce them. Observed
# on an 18-worker run: eighteen visible windows, the whole run crawling, and
# a trainer battle -- a few hundred frames of dialogue and animation, normally
# about a second -- taking minutes and looking like a hang. Compositing that
# many windows also competes with whatever else is using the GPU, which is
# the opposite of the intent on a shared machine.
#
# Spawned workers inherit this, since they re-read the environment when they
# re-import config from scratch. GIFs are unaffected: the screen buffer is
# still readable with no window.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from envs.forest_env import PokemonRedForestEnv  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from train_navigation_parallel import train  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "forest_q_table.json"
STATE_PATH = PROJECT_ROOT / "models" / "forest_parallel_state.json"

# The forest's shortest path to its real exit is 127 tiles (measured
# directly from the survey's own walkable-adjacency graph -- see
# rewards/forest_rewards.py), against Route 1's 35 and Route 2's 27, and
# a real maze rather than a corridor on top of that (~45% of its own
# bounding box), plus trainer battles along the way that cost real
# in-game turns even though they're a single env step each. 2000 is
# generous headroom over that shortest path without letting a failed
# episode run away entirely.
MAX_STEPS = 2000


def main():
    train(
        env_class=PokemonRedForestEnv,
        model_path=MODEL_PATH,
        state_path=STATE_PATH,
        gif_prefix="forest",
        max_steps=MAX_STEPS,
    )


if __name__ == "__main__":
    main()
