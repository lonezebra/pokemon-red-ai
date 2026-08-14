import os

# Visible by default here, unlike the training scripts -- the point of this
# one is to watch. POKEMON_AI_WINDOW_MODE=null makes it headless when what's
# wanted is the recorded GIF and the numbers rather than a live window.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "SDL2")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from core.atomic_io import write_json_atomic  # noqa: E402
from core.config import PROJECT_ROOT  # noqa: E402
from core.screen import save_gif  # noqa: E402
from envs.whole_game_env import PokemonRedWholeGameEnv  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "whole_game_ppo"
ROLLOUT_DIR = PROJECT_ROOT / "models" / "whole_game_rollouts"


def resolve_model(explicit):
    """A named checkpoint, else the newest thing training left behind.

    "Newest" means by mtime across *every* candidate, not just the most
    recently checked one. whole_game_latest.zip is only written when
    train_whole_game.py exits (finished or interrupted) -- during a live run
    it can be hours stale, or even left over from a previous run entirely,
    while CheckpointCallback keeps writing whole_game_<n>_steps.zip every
    100k steps. Checking latest.zip's existence first (as an earlier version
    of this did) would silently freeze anything watching a live run on
    whatever model existed before training started.
    """
    if explicit:
        return Path(explicit)

    candidates = list(MODEL_DIR.glob("whole_game_*_steps.zip"))
    latest = MODEL_DIR / "whole_game_latest.zip"
    if latest.exists():
        candidates.append(latest)

    if not candidates:
        raise FileNotFoundError(
            f"No trained model in {MODEL_DIR}. Run train_whole_game.py first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def play_episode(model, env, max_steps, deterministic, capture_frames):
    """
    One episode, recording the path taken and what the agent achieved.

    The per-step path is kept because it is what the visualisation actually
    draws (see render_whole_game_runs.py) and because aggregate scores hide
    the failure this project keeps rediscovering: a policy that racks up
    reward while going nowhere. Route 1's revisit loop looked fine in the
    numbers and was only caught by recording a run in full.
    """
    obs, _ = env.reset()

    path = []
    frames = []
    total_reward = 0.0
    components = {}
    info = {}

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        for name, value in info["reward_components"].items():
            components[name] = components.get(name, 0.0) + value

        path.append([info["map_id"], info["x"], info["y"]])

        if capture_frames and step % 4 == 0:
            frames.append(env.pyboy.screen.image.copy())

        if terminated or truncated:
            break

    return {
        "steps": len(path),
        "reward": total_reward,
        "components": components,
        "badges": info.get("badges", 0),
        "events": info.get("events", 0),
        "party_levels": info.get("party_levels", []),
        "tiles_explored": info.get("tiles_explored", 0),
        # The full walked path as [map_id, x, y] per step. Kept in step order
        # rather than as a set, because the mashup animation replays it --
        # where the agent went and in what order is the thing being drawn.
        "path": path,
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="checkpoint path (default: newest)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking the argmax")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model(args.model)
    print(f"Loading {model_path.name}")

    # Inference only, and on CPU deliberately: one environment's observations
    # are far too small a batch to be worth a round trip to the GPU.
    model = PPO.load(model_path, device="cpu")
    env = PokemonRedWholeGameEnv(max_steps=args.max_steps)

    ROLLOUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        for episode in range(1, args.episodes + 1):
            result = play_episode(
                model, env, args.max_steps,
                deterministic=not args.stochastic,
                capture_frames=not args.no_gif,
            )

            print(
                f"Episode {episode}: {result['badges']} badges, "
                f"{result['events']} events, "
                f"levels {result['party_levels']}, "
                f"{result['tiles_explored']} tiles, "
                f"reward {result['reward']:+.1f} over {result['steps']} steps"
            )

            frames = result.pop("frames")
            results.append(result)

            if frames and not args.no_gif:
                save_gif(frames, f"whole_game_episode{episode:02d}.gif")

    finally:
        env.close()

    # Ranked by the game's own measures first and the reward function second.
    # If those two ever disagree the reward needs fixing, and that is only
    # visible when both are recorded.
    best = max(results, key=lambda r: (r["badges"], r["events"],
                                       r["tiles_explored"]))

    print()
    print("=" * 60)
    print(f"{len(results)} episodes")
    print(f"  best:   {best['badges']} badges, {best['events']} events, "
          f"{best['tiles_explored']} tiles")
    print(f"  mean tiles explored: "
          f"{np.mean([r['tiles_explored'] for r in results]):.0f}")
    print(f"  mean reward:         "
          f"{np.mean([r['reward'] for r in results]):+.1f}")
    print("=" * 60)

    rollout_path = ROLLOUT_DIR / f"{model_path.stem}_rollouts.json"
    write_json_atomic(rollout_path, {
        "model": model_path.name,
        "deterministic": not args.stochastic,
        "episodes": results,
    })
    print(f"Rollouts written to {rollout_path}")


if __name__ == "__main__":
    main()
