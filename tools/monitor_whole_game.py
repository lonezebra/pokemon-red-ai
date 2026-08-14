"""
Print a periodic progress report while train_whole_game.py runs elsewhere,
so watching a live run doesn't mean re-running watch_whole_game.py and
render_whole_game_runs.py by hand every fifteen minutes.

Each round: load whatever checkpoint is newest (watch_whole_game.resolve_model
-- the same logic the manual eval command uses, not reimplemented here),
play a few episodes, and print the phase signals called out when this track
started: does it escape Oak's Lab, does it get a starter, does story
progress climb, does a badge ever land.

Deliberately reuses watch_whole_game.play_episode rather than a lighter
custom rollout loop, so what this prints during a live run and what
watch_whole_game.py prints when run by hand can never quietly disagree.

    cd src && ../.venv/bin/python3 ../tools/monitor_whole_game.py
    cd src && ../.venv/bin/python3 ../tools/monitor_whole_game.py --interval 600 --episodes 5
    cd src && ../.venv/bin/python3 ../tools/monitor_whole_game.py --once
"""

import argparse
import collections
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stable_baselines3 import PPO  # noqa: E402

from envs.whole_game_env import PokemonRedWholeGameEnv  # noqa: E402
from watch_whole_game import play_episode, resolve_model  # noqa: E402

# Reported in this order because it's the order the milestones actually
# happen in -- printing "badges" before "levels" would read strangely for a
# run that hasn't gotten a starter yet.
MAP_NAMES = {
    38: "bedroom", 37: "house (downstairs)", 0: "Pallet Town",
    40: "Oak's Lab", 39: "rival's house", 12: "Route 1", 1: "Viridian City",
    51: "Viridian Forest", 13: "Route 2", 2: "Pewter City", 54: "Pewter Gym",
    14: "Route 3",
}

PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def timestamp():
    return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d %I:%M:%S %p %Z")


def training_log_tail(log_path):
    """The last fps/timesteps/reward line SB3 wrote, if a log was given.
    Best-effort: a monitor for a run in another terminal shouldn't die
    because that terminal's log path doesn't exist yet or was never given."""
    if not log_path or not Path(log_path).exists():
        return None

    lines = Path(log_path).read_text(errors="replace").splitlines()
    wanted = ("fps", "total_timesteps", "ep_rew_mean", "ep_len_mean")
    recent = [
        line.strip() for line in lines[-200:]
        if any(key in line for key in wanted)
    ]
    return recent[-4:] if recent else None


def run_round(episodes, max_steps):
    model_path = resolve_model(None)
    model = PPO.load(model_path, device="cpu")
    env = PokemonRedWholeGameEnv(max_steps=max_steps)

    results = []
    try:
        for _ in range(episodes):
            results.append(play_episode(
                model, env, max_steps, deterministic=False,
                capture_frames=False,
            ))
    finally:
        env.close()

    map_counts = collections.Counter()
    for result in results:
        for map_id, x, y in result["path"]:
            map_counts[map_id] += 1
    total_steps = sum(map_counts.values()) or 1

    return model_path, results, map_counts, total_steps


def print_report(model_path, results, map_counts, total_steps, log_path):
    print("=" * 68)
    print(f"[{timestamp()}] {model_path.name}")
    print("=" * 68)

    tail = training_log_tail(log_path)
    if tail:
        print("Training:")
        for line in tail:
            print(f"  {line}")
        print()

    best = max(results, key=lambda r: (r["badges"], r["events"],
                                       r["tiles_explored"]))
    print(f"Eval ({len(results)} episodes, stochastic):")
    print(f"  badges:   best {best['badges']}")
    print(f"  events:   best {best['events']}  "
          f"(all: {[r['events'] for r in results]})")
    print(f"  levels:   {[r['party_levels'] for r in results]}")
    print(f"  tiles:    mean {sum(r['tiles_explored'] for r in results) / len(results):.0f}")
    print(f"  reward:   mean {sum(r['reward'] for r in results) / len(results):+.1f}")

    print()
    print("Where the steps went:")
    for map_id, count in map_counts.most_common(6):
        name = MAP_NAMES.get(map_id, f"map {map_id}")
        print(f"  {name:<22} {100 * count / total_steps:5.1f}%  ({count} steps)")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=900,
                        help="seconds between rounds (default 900 = 15 min)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--log", default="/tmp/whole_game.log",
                        help="training stdout log to tail, if any")
    parser.add_argument("--once", action="store_true",
                        help="print one report and exit, rather than loop")
    args = parser.parse_args()

    print(f"Watching {os.environ.get('MODEL_DIR', 'models/whole_game_ppo')} "
          f"every {args.interval}s. Ctrl-C to stop.\n")

    while True:
        try:
            model_path, results, map_counts, total_steps = run_round(
                args.episodes, args.max_steps
            )
            print_report(model_path, results, map_counts, total_steps, args.log)
        except FileNotFoundError as error:
            print(f"[{timestamp()}] {error}")

        if args.once:
            break

        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


if __name__ == "__main__":
    main()
