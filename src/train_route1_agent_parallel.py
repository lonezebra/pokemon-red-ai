import json
import multiprocessing as mp

from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from core.screen import save_gif
from train_route1_agent import run_demo_episode, WARM_START_EPSILON

# Parallel version of train_route1_agent.py's single-process loop: this
# container has 4 CPU cores and training was only ever using one of
# them. Each worker is a fully independent PyBoy instance training its
# own copy of the Q-table in its own process (real parallelism, not
# threads -- PyBoy is CPU-bound C code, threads wouldn't help past the
# GIL). Naively running N copies of train_route1_agent.py at once would
# have them all overwrite the same shared Q-table file with no
# coordination, silently discarding whichever worker's save loses the
# race. Instead, each round: every worker loads the current shared
# table, trains EPISODES_PER_ROUND episodes independently, and saves to
# its own file; the driver then merges all worker tables by averaging
# every (state, action) value the workers actually have an opinion on,
# and that merged table becomes the next round's shared starting point.

MODEL_PATH = PROJECT_ROOT / "models" / "route1_q_table.json"
LEGACY_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "route1_checkpoint.json"
STATE_PATH = PROJECT_ROOT / "models" / "route1_parallel_state.json"
WORKER_DIR = PROJECT_ROOT / "models" / "parallel_workers"

NUM_WORKERS = 3  # leaves 1 of this container's 4 cores free
EPISODES_PER_ROUND = 100
MAX_STEPS = 800
MAX_ROUNDS = 500  # a generous cap, not an expected stopping point
EPSILON_MIN = 0.05
EPSILON_DECAY_PER_EPISODE = 0.998


def run_worker(episodes, epsilon, shared_table_path, output_table_path, output_summary_path, max_steps):
    env = PokemonRedRoute1Env(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())

    if shared_table_path.exists():
        agent.load(shared_table_path)
    agent.epsilon = epsilon

    successes = 0

    for _ in range(episodes):
        obs = env.reset()
        info = {}

        for _ in range(max_steps):
            action = agent.choose_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs

            if done:
                break

        agent.decay_epsilon()

        if info.get("reached_goal"):
            successes += 1

    env.close()

    agent.save(output_table_path)
    with open(output_summary_path, "w") as f:
        json.dump({"successes": successes, "episodes": episodes}, f)


def merge_tables(worker_table_paths, output_path):
    accumulated = {}

    for path in worker_table_paths:
        with open(path) as f:
            table = json.load(f)

        for key, values in table.items():
            if key not in accumulated:
                accumulated[key] = [[] for _ in values]
            for i, value in enumerate(values):
                accumulated[key][i].append(value)

    merged = {
        key: [sum(values) / len(values) for values in per_action]
        for key, per_action in accumulated.items()
    }

    with open(output_path, "w") as f:
        json.dump(merged, f)


def save_state(round_num, epsilon, total_episodes, successes, best_demo_key):
    with open(STATE_PATH, "w") as f:
        json.dump(
            {
                "round": round_num,
                "epsilon": epsilon,
                "total_episodes": total_episodes,
                "successes": successes,
                "best_demo_key": list(best_demo_key) if best_demo_key is not None else None,
            },
            f,
        )


def load_state():
    if not STATE_PATH.exists():
        return None

    with open(STATE_PATH) as f:
        data = json.load(f)

    if data["best_demo_key"] is not None:
        data["best_demo_key"] = tuple(data["best_demo_key"])

    return data


def initial_state():
    """
    Figures out where to start from: resuming a previous parallel run,
    continuing a single-process run that got stopped mid-batch (its
    checkpoint's epsilon carries over rather than jumping back up),
    warm-starting from a completed single-process Q-table, or a
    completely fresh start.
    """

    state = load_state()
    if state is not None:
        print(f"Resuming parallel training from round {state['round']}, epsilon={state['epsilon']:.3f}")
        return state["round"] + 1, state["epsilon"], state["total_episodes"], state["successes"], state["best_demo_key"]

    if LEGACY_CHECKPOINT_PATH.exists():
        with open(LEGACY_CHECKPOINT_PATH) as f:
            legacy = json.load(f)
        print(f"Starting parallel training, continuing single-process run at epsilon={legacy['epsilon']:.3f}")
        LEGACY_CHECKPOINT_PATH.unlink()
        return 1, legacy["epsilon"], 0, 0, None

    if MODEL_PATH.exists():
        print(f"Starting parallel training, warm-started from existing Q-table, epsilon={WARM_START_EPSILON:.3f}")
        return 1, WARM_START_EPSILON, 0, 0, None

    print("Starting parallel training from scratch")
    return 1, 1.0, 0, 0, None


def main():
    WORKER_DIR.mkdir(parents=True, exist_ok=True)

    start_round, epsilon, total_episodes, successes, best_demo_key = initial_state()

    for round_num in range(start_round, MAX_ROUNDS + 1):
        worker_table_paths = [WORKER_DIR / f"worker{i}.json" for i in range(NUM_WORKERS)]
        worker_summary_paths = [WORKER_DIR / f"worker{i}_summary.json" for i in range(NUM_WORKERS)]

        processes = [
            mp.Process(
                target=run_worker,
                args=(EPISODES_PER_ROUND, epsilon, MODEL_PATH, worker_table_paths[i], worker_summary_paths[i], MAX_STEPS),
            )
            for i in range(NUM_WORKERS)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        merge_tables(worker_table_paths, MODEL_PATH)

        round_successes = 0
        for path in worker_summary_paths:
            with open(path) as f:
                round_successes += json.load(f)["successes"]

        successes += round_successes
        total_episodes += EPISODES_PER_ROUND * NUM_WORKERS
        epsilon = max(EPSILON_MIN, epsilon * (EPSILON_DECAY_PER_EPISODE ** EPISODES_PER_ROUND))

        print(
            f"Round {round_num:3d}  total_episodes={total_episodes}  epsilon={epsilon:.3f}  "
            f"successes this round: {round_successes}/{NUM_WORKERS * EPISODES_PER_ROUND}  "
            f"cumulative successes: {successes}/{total_episodes}"
        )

        demo_env = PokemonRedRoute1Env(max_steps=MAX_STEPS)
        demo_agent = QLearningAgent(num_actions=num_actions())
        demo_agent.load(MODEL_PATH)
        demo = run_demo_episode(demo_env, demo_agent, MAX_STEPS)
        demo_env.close()

        save_gif(demo["frames"], f"route1_progress_round{round_num:03d}.gif")
        print(
            f"  [demo] reached_goal={demo['reached_goal']} "
            f"tiles_visited={demo['tiles_visited']} steps={demo['steps']}"
        )

        demo_key = (demo["reached_goal"], demo["tiles_visited"], -demo["steps"])
        if best_demo_key is None or demo_key > best_demo_key:
            best_demo_key = demo_key
            save_gif(demo["frames"], "route1_best_so_far.gif")
            print(f"  [demo] new best so far (round {round_num})")

        save_state(round_num, epsilon, total_episodes, successes, best_demo_key)

    print()
    print(f"Finished {MAX_ROUNDS} rounds. Total successes: {successes}/{total_episodes}")


if __name__ == "__main__":
    main()
