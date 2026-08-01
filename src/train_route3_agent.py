import os

# Headless unless explicitly asked otherwise, and this must run before any
# core import, because core/config.py reads the variable at import time.
#
# The default is "SDL2", which opens a real Game Boy window per emulator.
# That is what you want for watch_*.py, and actively harmful here -- see
# train_forest_agent.py for the measured cost on a many-worker run.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from envs.route3_env import PokemonRedRoute3Env  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from train_navigation_parallel import train  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "route3_q_table.json"
STATE_PATH = PROJECT_ROOT / "models" / "route3_parallel_state.json"

# GOAL_TILE is 30 hops from the entrance by shortest path (see
# rewards/route3_rewards.py) -- shorter than the forest's 127, but not a
# corridor and not without forced fights, so this sits between Route
# 1/2's 600-800 and the forest's 2000: generous headroom over the
# shortest path plus however many trainer fights an exploring policy
# triggers along the way, without letting a failed episode run away
# entirely.
MAX_STEPS = 1000


def main():
    train(
        env_class=PokemonRedRoute3Env,
        model_path=MODEL_PATH,
        state_path=STATE_PATH,
        gif_prefix="route3",
        max_steps=MAX_STEPS,
    )


if __name__ == "__main__":
    main()
