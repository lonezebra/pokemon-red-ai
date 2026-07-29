from envs.forest_env import PokemonRedForestEnv
from core.config import PROJECT_ROOT
from train_navigation_parallel import train

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
