import json
from datetime import datetime

from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "route1_q_table.json"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "route1_checkpoint.json"
MASHUP_DIR = PROJECT_ROOT / "screenshots" / "mashups"


def default_run_label():
    # Prefer naming the folder after how far training had actually gotten
    # (e.g. "ep0500") so runs from different training milestones are easy
    # to tell apart at a glance -- fall back to a timestamp if training
    # already finished and cleaned up its checkpoint (see
    # train_route1_agent.py's end-of-run cleanup).
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            episode = json.load(f)["episode"]
        return f"ep{episode:04d}"

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_rollout(env, agent, max_steps):
    obs = env.reset()
    positions = [(obs["x"], obs["y"])]
    reached_goal = False

    for _ in range(max_steps):
        action = agent.choose_action(obs, greedy=True)
        obs, _, done, info = env.step(action)
        positions.append((obs["x"], obs["y"]))

        if done:
            reached_goal = info.get("reached_goal", False)
            break

    return positions, reached_goal


def main(num_runs=150, max_steps=300, run_label=None):
    # greedy (not epsilon-random) action selection, same as
    # watch_route1_agent.py -- this is meant to show what the agent has
    # actually learned so far, not force extra exploration. Early in
    # training the Q-table is still mostly empty/all-zero for unvisited
    # states, so greedy ties are broken randomly anyway (see
    # QLearningAgent.choose_action), which is exactly why 150 runs still
    # produce visibly different paths instead of one line repeated 150
    # times.
    #
    # max_steps is capped lower than training's 800 -- this is purely for
    # keeping the mashup video-length/runtime reasonable, not related to
    # the actual training step budget.
    env = PokemonRedRoute1Env(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(MODEL_PATH)

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

    output_path = run_dir / "route1_mashup_rollouts.json"
    with open(output_path, "w") as f:
        json.dump({"runs": runs, "max_steps": max_steps}, f)

    print()
    print(f"Total: {successes}/{num_runs} reached Viridian City")
    print(f"Saved {len(runs)} rollouts to {output_path}")


if __name__ == "__main__":
    main()
