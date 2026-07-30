import json
import math
import os
import multiprocessing as mp

from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from core.screen import save_gif

# Route-agnostic version of train_route1_agent_parallel.py, which was
# written for Route 1 specifically before there was a second navigation
# task to share it with. Everything route-specific -- the environment,
# where the Q-table lives, what the demo GIFs are called -- is a
# parameter here, so Route 2 (and Viridian Forest, and whatever follows)
# gets the same parallel training without another copy of the loop.
#
# The parallelism itself works the way it did for Route 1: this container
# has 4 CPU cores, and PyBoy is CPU-bound C code, so workers are real
# processes rather than threads. Naively running N trainers at once would
# have them all overwrite one shared Q-table with no coordination,
# silently discarding whichever save lost the race. Instead each round
# gives every worker the same starting table, lets them train
# independently into their own files, then merges by averaging every
# (state, action) value the workers actually formed an opinion on.

# 'spawn', not this platform's 'fork' default -- Route 1/Route 2's own
# envs never loaded a PyTorch model, so fork was never risky for them,
# but core/parallel_survey.py hit a real, sustained-load-only hang from
# forking a process that had already initialized PyTorch's own thread
# pool (via the driver merely importing stable_baselines3 to reference a
# model-loading function, not even calling it). Any env this trains that
# loads a model internally -- Viridian Forest's, which auto-resolves
# forced trainer battles with the already-solved trainer-battle DQN --
# hits the exact same hazard, so it's fixed here too rather than waiting
# to rediscover it on a long run.
_SPAWN_CTX = mp.get_context("spawn")

# One worker per core, rather than the 4 this was written against, so
# the same code uses whatever machine it lands on. POKEMON_RED_WORKERS
# overrides it.
#
# Raising this trades two things off, and they pull in opposite
# directions. More workers means more episodes per round, but the merge
# averages every worker's Q-table together, and averaging more
# independently-diverged policies makes the merged greedy policy less
# coherent, not more -- so on a high-core machine, lowering
# episodes_per_round to keep rounds short is usually a better use of the
# cores than leaving it at 100 and quadrupling each round's length.
# Memory is the other cap: each worker is a full PyBoy plus, for the
# forest, its own copy of the trainer-battle DQN, ~700MB resident.
DEFAULT_NUM_WORKERS = int(os.environ.get("POKEMON_RED_WORKERS") or (os.cpu_count() or 4))
DEFAULT_EPISODES_PER_ROUND = 100

# See core/parallel_survey.py for why: PyBoy is single-threaded and is
# the bottleneck, so one compute thread per worker is what's wanted.
# Without this, torch gives each worker one thread per core, and the
# resulting oversubscription gets worse the more cores the machine has.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
DEFAULT_MAX_ROUNDS = 500  # a generous cap, not an expected stopping point

WARM_START_EPSILON = 0.3
EPSILON_MIN = 0.05
EPSILON_DECAY_PER_EPISODE = 0.998

DEMO_MAX_FRAMES = 300
WORKER_DIR = PROJECT_ROOT / "models" / "parallel_workers"


def run_demo_episode(env, agent, max_steps):
    """
    One greedy (no exploration) episode, capturing a frame per step, so
    progress can actually be watched -- this project runs headless, so a
    stitched GIF is the closest thing to looking over its shoulder.
    Subsampled if the episode runs long, so a slow early episode doesn't
    produce an enormous file.
    """

    obs = env.reset()
    frames = [env.pyboy.screen.image.copy()]
    info = {}

    for _ in range(max_steps):
        action = agent.choose_action(obs, greedy=True)
        obs, _, done, info = env.step(action)
        frames.append(env.pyboy.screen.image.copy())
        if done:
            break

    if len(frames) > DEMO_MAX_FRAMES:
        stride = math.ceil(len(frames) / DEMO_MAX_FRAMES)
        frames = frames[::stride]

    return {
        "frames": frames,
        "reached_goal": info.get("reached_goal", False),
        "tiles_visited": len(env.visited_positions),
        "steps": info.get("step_count", 0),
    }


def run_worker(env_class, episodes, epsilon, shared_table_path,
               output_table_path, output_summary_path, max_steps):
    env = env_class(max_steps=max_steps)
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


def save_progress(state_path, round_num, epsilon, total_episodes, successes, best_demo_key):
    with open(state_path, "w") as f:
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


def load_progress(state_path):
    if not state_path.exists():
        return None

    with open(state_path) as f:
        data = json.load(f)

    if data["best_demo_key"] is not None:
        data["best_demo_key"] = tuple(data["best_demo_key"])

    return data


def initial_state(state_path, model_path):
    progress = load_progress(state_path)
    if progress is not None:
        print(
            f"Resuming from round {progress['round']}, "
            f"epsilon={progress['epsilon']:.3f}, "
            f"successes so far: {progress['successes']}/{progress['total_episodes']}"
        )
        return (
            progress["round"] + 1,
            progress["epsilon"],
            progress["total_episodes"],
            progress["successes"],
            progress["best_demo_key"],
        )

    if model_path.exists():
        print(f"Warm-starting from existing Q-table, epsilon={WARM_START_EPSILON:.3f}")
        return 1, WARM_START_EPSILON, 0, 0, None

    print("Starting from scratch")
    return 1, 1.0, 0, 0, None


def train(
    env_class,
    model_path,
    state_path,
    gif_prefix,
    max_steps,
    num_workers=DEFAULT_NUM_WORKERS,
    episodes_per_round=DEFAULT_EPISODES_PER_ROUND,
    max_rounds=DEFAULT_MAX_ROUNDS,
):
    worker_dir = WORKER_DIR / gif_prefix
    worker_dir.mkdir(parents=True, exist_ok=True)

    start_round, epsilon, total_episodes, successes, best_demo_key = initial_state(
        state_path, model_path
    )

    for round_num in range(start_round, max_rounds + 1):
        table_paths = [worker_dir / f"worker{i}.json" for i in range(num_workers)]
        summary_paths = [worker_dir / f"worker{i}_summary.json" for i in range(num_workers)]

        processes = [
            _SPAWN_CTX.Process(
                target=run_worker,
                args=(
                    env_class,
                    episodes_per_round,
                    epsilon,
                    model_path,
                    table_paths[i],
                    summary_paths[i],
                    max_steps,
                ),
            )
            for i in range(num_workers)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

        merge_tables(table_paths, model_path)

        round_successes = 0
        for path in summary_paths:
            with open(path) as f:
                round_successes += json.load(f)["successes"]

        successes += round_successes
        total_episodes += episodes_per_round * num_workers
        epsilon = max(EPSILON_MIN, epsilon * (EPSILON_DECAY_PER_EPISODE ** episodes_per_round))

        print(
            f"Round {round_num:3d}  total_episodes={total_episodes}  epsilon={epsilon:.3f}  "
            f"successes this round: {round_successes}/{num_workers * episodes_per_round}  "
            f"cumulative: {successes}/{total_episodes}"
        )

        demo_env = env_class(max_steps=max_steps)
        demo_agent = QLearningAgent(num_actions=num_actions())
        demo_agent.load(model_path)
        demo = run_demo_episode(demo_env, demo_agent, max_steps)
        demo_env.close()

        save_gif(demo["frames"], f"{gif_prefix}_progress_round{round_num:03d}.gif")
        print(
            f"  [demo] reached_goal={demo['reached_goal']} "
            f"tiles_visited={demo['tiles_visited']} steps={demo['steps']}"
        )

        demo_key = (demo["reached_goal"], demo["tiles_visited"], -demo["steps"])
        if best_demo_key is None or demo_key > best_demo_key:
            best_demo_key = demo_key
            save_gif(demo["frames"], f"{gif_prefix}_best_so_far.gif")
            print(f"  [demo] new best so far (round {round_num})")

        save_progress(state_path, round_num, epsilon, total_episodes, successes, best_demo_key)

    print()
    print(f"Finished {max_rounds} rounds. Total successes: {successes}/{total_episodes}")
