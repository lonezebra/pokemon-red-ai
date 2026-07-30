"""
Record forest navigation rollouts for the mashup renderer.

    cd src && ../.venv/bin/python3 generate_forest_mashup_rollouts.py

The forest counterpart of generate_route1_mashup_rollouts.py, and the
half of the mashup pipeline that had to be new -- the renderer
(render_route1_mashup.py) was already parameterized by map prefix and
rollout file, so it draws these on the forest panorama unchanged.

Differences from Route 1 worth noting:

  - Positions are recorded only while the player is on the forest map.
    Reaching the goal means stepping off it (onto the map-47 connector),
    and a loss means blacking out to a Pokemon Center; either way the
    panorama has no tile for the destination, so the trace simply ends
    at the last forest tile, and reached_goal colors it.

  - The same MASHUP_EPSILON=0.15 trick applies, for the same measured
    reason: a deterministic policy in a deterministic env produced 150
    bit-identical Route 1 rollouts, which made the mashup pointless.
    Slight exploration keeps runs mostly on-policy but visually distinct.

  - max_steps is 300 against a 127-hop shortest path: enough slack for
    epsilon detours, short enough to keep the GIF watchable. Runs that
    time out render in the unfinished color, which is honest -- a mashup
    generated before training converges *should* look mostly red.
"""

import json
from datetime import datetime

from envs.forest_env import PokemonRedForestEnv
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from rewards.forest_rewards import FOREST_MAP_ID

MODEL_PATH = PROJECT_ROOT / "models" / "forest_q_table.json"
STATE_PATH = PROJECT_ROOT / "models" / "forest_parallel_state.json"
MASHUP_DIR = PROJECT_ROOT / "screenshots" / "mashups"

MASHUP_EPSILON = 0.15


def default_run_label():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            round_num = json.load(f)["round"]
        return f"forest_round{round_num:03d}"
    return "forest_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def run_rollout(env, agent, max_steps):
    obs = env.reset()
    positions = [(obs["x"], obs["y"])]
    reached_goal = False

    for _ in range(max_steps):
        action = agent.choose_action(obs, greedy=False)
        obs, _, done, info = env.step(action)
        if obs["map_id"] == FOREST_MAP_ID:
            positions.append((obs["x"], obs["y"]))
        if done:
            reached_goal = info.get("reached_goal", False)
            break

    return positions, reached_goal


def main(num_runs=150, max_steps=300, run_label=None):
    env = PokemonRedForestEnv(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(MODEL_PATH)
    agent.epsilon = MASHUP_EPSILON

    runs = []
    successes = 0

    for run_idx in range(1, num_runs + 1):
        positions, reached_goal = run_rollout(env, agent, max_steps)
        runs.append({"positions": positions, "reached_goal": reached_goal})
        if reached_goal:
            successes += 1
        if run_idx % 10 == 0:
            print(f"Run {run_idx:3d}/{num_runs}  successes so far: {successes}/{run_idx}")

    env.close()

    run_label = run_label or default_run_label()
    run_dir = MASHUP_DIR / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "forest_mashup_rollouts.json"
    with open(output_path, "w") as f:
        json.dump({"runs": runs, "max_steps": max_steps}, f)

    print()
    print(f"Total: {successes}/{num_runs} reached the Pewter-side exit")
    print(f"Saved {len(runs)} rollouts to {output_path}")
    return run_label


if __name__ == "__main__":
    main()
