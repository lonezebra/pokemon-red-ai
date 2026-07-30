import json
import math
import os
import multiprocessing as mp

from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.atomic_io import write_json_atomic
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

# Episodes per *round*, shared across all workers rather than assigned to
# each -- see run_worker for why claiming from a shared budget beats
# handing every worker an equal share.
#
# Because this is the round's total, extra cores shorten rounds instead of
# enlarging them. A round is a barrier: nobody sees anyone else's learning
# until every worker stops and the driver merges. If more workers each ran
# a fixed share, rounds would stay the same length and simply contain more
# experience per merge; keeping the total fixed instead means more workers
# drain it proportionally faster, so the same epsilon-per-episode schedule
# yields more merge points and returns feedback sooner.
#
# Shorter rounds also push against the divergence actually observed here:
# training success climbed steadily (0.5% -> 11.5% over four rounds) while
# the greedy demo regressed (117 -> 59 tiles visited). Averaging
# independently-diverged Q-tables produces an argmax that need not be good
# in any of their directions, and the longer workers run between merges,
# the further they drift before being averaged.
#
# 400 keeps the container's own numbers comparable: it ran 4 workers x 100
# episodes, which is the same round total.
DEFAULT_EPISODES_PER_ROUND = int(
    os.environ.get("POKEMON_RED_EPISODES_PER_ROUND") or 400
)

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


def run_worker(env_class, remaining, epsilon, shared_table_path,
               output_table_path, output_summary_path, max_steps):
    """
    Run episodes claimed from a budget shared with every other worker in
    the round, until it is exhausted.

    Workers used to each be handed an identical episode count, which is
    only efficient when every worker runs at the same speed. A round is a
    barrier -- the driver joins all of them before merging -- so with
    equal counts the round finishes at the pace of the slowest worker and
    faster ones sit idle at the end. On a machine whose cores are
    deliberately not identical (Apple Silicon ships two or more
    performance tiers, and a 6-super/12-performance split means two
    thirds of the pool runs a different speed) that idle tail is
    structural rather than incidental.

    Claiming from a shared counter instead makes the split self-balancing:
    a faster core simply completes more episodes, every worker stops when
    the budget is gone, and the round's tail shrinks to at most one
    in-flight episode regardless of how uneven the cores are. It also
    means the round's total is what's actually configured, rather than
    per-worker-count times workers.

    Each worker still reports how many episodes it ran, since that's no
    longer inferable and the driver needs the real total for its epsilon
    schedule and its success accounting.
    """
    env = env_class(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())

    if shared_table_path.exists():
        agent.load(shared_table_path)
    agent.epsilon = epsilon

    successes = 0
    episodes_run = 0

    while True:
        # Claim one episode. The lock is held only for the decrement, not
        # for the episode itself, and episodes take seconds at minimum, so
        # contention is irrelevant even with many workers.
        with remaining.get_lock():
            if remaining.value <= 0:
                break
            remaining.value -= 1

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
        episodes_run += 1
        if info.get("reached_goal"):
            successes += 1

    env.close()

    agent.save(output_table_path)
    write_json_atomic(
        output_summary_path, {"successes": successes, "episodes": episodes_run}
    )


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

    write_json_atomic(output_path, merged)


def save_progress(state_path, round_num, epsilon, total_episodes, successes, best_demo_key):
    # Atomic, and written last in a round on purpose: this is the file
    # tools/checkpoint_artifacts.sh watches to decide a round finished, so
    # it must never appear complete while the Q-table beside it isn't.
    write_json_atomic(
        state_path,
        {
            "round": round_num,
            "epsilon": epsilon,
            "total_episodes": total_episodes,
            "successes": successes,
            "best_demo_key": list(best_demo_key) if best_demo_key is not None else None,
        },
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

        # The round's episode budget, claimed one at a time by whichever
        # worker is free. Created from the spawn context so the shared
        # lock survives being passed to spawned children.
        remaining = _SPAWN_CTX.Value("i", episodes_per_round)

        processes = [
            _SPAWN_CTX.Process(
                target=run_worker,
                args=(
                    env_class,
                    remaining,
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
        round_episodes = 0
        episodes_by_worker = []
        for path in summary_paths:
            with open(path) as f:
                summary = json.load(f)
            round_successes += summary["successes"]
            round_episodes += summary["episodes"]
            episodes_by_worker.append(summary["episodes"])

        successes += round_successes
        total_episodes += round_episodes

        # Each worker decays its own epsilon once per episode it ran, so
        # the round's effective decay is over the *average* worker's
        # episode count, not the round total. Under the old equal-shares
        # scheme those were the same number; with a shared budget they
        # aren't, and using the total here would collapse epsilon roughly
        # num_workers times too fast.
        episodes_per_worker = round_episodes / num_workers if num_workers else 0
        epsilon = max(
            EPSILON_MIN, epsilon * (EPSILON_DECAY_PER_EPISODE ** episodes_per_worker)
        )

        # The spread across workers is the diagnostic for whether the
        # cores are as uneven as expected -- on identical cores it should
        # be nearly flat, and on a mixed-tier machine the faster tier
        # should visibly claim more.
        spread = (
            f"  per-worker: {min(episodes_by_worker)}-{max(episodes_by_worker)}"
            if episodes_by_worker and min(episodes_by_worker) != max(episodes_by_worker)
            else ""
        )
        print(
            f"Round {round_num:3d}  total_episodes={total_episodes}  epsilon={epsilon:.3f}  "
            f"successes this round: {round_successes}/{round_episodes}  "
            f"cumulative: {successes}/{total_episodes}{spread}"
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
