import json
import math
import os
import random
import signal
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


DEMO_STUCK_LIMIT = 8


def run_demo_episode(env, agent, max_steps):
    """
    One near-greedy episode, capturing a frame per step, so progress can
    actually be watched -- this project runs headless, so a stitched GIF is
    the closest thing to looking over its shoulder. Subsampled if the
    episode runs long, so a slow early episode doesn't produce an enormous
    file.

    Near-greedy rather than purely greedy, because a purely greedy demo
    deadlocks by construction. The observation is exactly (map_id, x, y),
    so if the Q-table's argmax at some tile points into a wall or a
    trainer, the step doesn't move, the next observation is *identical*,
    the argmax is therefore identical, and the episode spends its entire
    step budget bumping one tile. Nothing breaks the cycle: unlike
    training, a demo performs no updates, so the action's value never
    drops. This is what "all the workers are stuck on a trainer" looked
    like, and it also explains demos reporting steps=2000 with a handful
    of tiles visited while the training success rate was climbing --
    training uses epsilon-greedy and escapes on its own.

    After DEMO_STUCK_LIMIT steps without the position changing, one random
    action is taken to break out. The count of those nudges is returned,
    and it is the more useful number of the two: a policy that reaches the
    goal with zero nudges is genuinely solving the maze, while one that
    needs dozens is being carried by them. Reporting it keeps the
    loop-breaker from quietly flattering the agent.
    """

    obs = env.reset()
    frames = [env.pyboy.screen.image.copy()]
    info = {}

    # Seeded per demo so a round's demo is reproducible from its Q-table.
    rng = random.Random(0)
    stuck_for = 0
    nudges = 0

    for _ in range(max_steps):
        if stuck_for >= DEMO_STUCK_LIMIT:
            action = rng.randrange(num_actions())
            nudges += 1
            stuck_for = 0
        else:
            action = agent.choose_action(obs, greedy=True)

        previous_obs = obs
        obs, _, done, info = env.step(action)
        frames.append(env.pyboy.screen.image.copy())

        stuck_for = stuck_for + 1 if obs == previous_obs else 0

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
        "nudges": nudges,
    }


_STOP_REQUESTED = False
_CURRENT_BUDGET = None


def _request_graceful_stop(signum, frame):
    """
    Turn Ctrl-C into "finish this round, save it, then exit".

    Interrupting used to throw the whole in-flight round away: the merge
    and save_progress only run once every worker has finished, so a round
    killed partway through left nothing behind. With rounds lasting from
    forty minutes to a few hours, that made stopping training to free the
    machine up genuinely expensive -- which is the normal thing to want on
    a machine that is also used for other things.

    Draining the shared budget is all a graceful stop needs, which is a
    property the queue gets for free: workers check the budget between
    episodes, so zeroing it makes each of them finish the episode it is on
    and exit normally. The driver then merges and saves exactly as it
    would at the end of any round, and stops instead of starting another.
    Worst case the interrupt costs one episode per worker rather than a
    whole round.

    A second Ctrl-C escalates to the old behavior, so a wedged episode
    can't trap someone who genuinely needs the cores back now.
    """
    global _STOP_REQUESTED

    if _STOP_REQUESTED:
        raise KeyboardInterrupt("second interrupt: abandoning the round")

    _STOP_REQUESTED = True

    if _CURRENT_BUDGET is not None:
        with _CURRENT_BUDGET.get_lock():
            _CURRENT_BUDGET.value = 0

    print(
        "\nStopping: letting workers finish the episodes they're on, then "
        "merging and saving this round before exiting. Ctrl-C again to "
        "abandon the round instead."
    )


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
    # The terminal delivers Ctrl-C to the whole foreground process group,
    # so without this every worker would die on the signal before it could
    # save what it had learned -- defeating the graceful stop entirely. The
    # driver owns the interrupt and coordinates shutdown by draining the
    # budget; workers only need to keep checking it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    env = env_class(max_steps=max_steps)
    agent = QLearningAgent(num_actions=num_actions())

    if shared_table_path.exists():
        agent.load(shared_table_path)
    agent.epsilon = epsilon

    successes = 0
    episodes_run = 0
    update_counts = {}

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
            # Serialized exactly as QLearningAgent.save serializes keys,
            # with the action appended, so the merge can line counts up
            # with table entries without re-parsing anything.
            key = f"{obs['map_id']},{obs['x']},{obs['y']}|{action}"
            update_counts[key] = update_counts.get(key, 0) + 1
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
    write_json_atomic(str(output_table_path) + ".counts", update_counts)


def merge_tables(worker_table_paths, output_path):
    """
    Visit-weighted average of the workers' Q-tables.

    A plain average has a dilution problem that grows with the worker
    count. Every worker starts a round from the same shared table, so a
    worker that never visited some state finishes the round still holding
    the inherited value -- and contributes it to the average with the same
    weight as the one worker that actually learned something there. At 18
    workers, a value only one worker updated moves toward its new estimate
    at 1/18th of the intended rate. Observed directly: five consecutive
    greedy demos frozen at exactly 115 tiles, nudges=0, while training
    successes climbed from 10 to 34 per round -- the well-visited shallow
    region kept learning while the deep-maze frontier was averaged back
    into place every merge.

    Each worker therefore reports how many times it updated each
    (state, action) this round, and the merge weights contributions by
    those counts, so untouched inherited copies contribute nothing. Where
    nobody updated a value at all, it carries through unchanged (every
    copy is the same inherited number). A worker table without a counts
    sidecar falls back to weight 1 on every entry, which is exactly the
    old behavior.
    """
    accumulated = {}

    for path in worker_table_paths:
        with open(path) as f:
            table = json.load(f)
        try:
            with open(str(path) + ".counts") as f:
                counts = json.load(f)
        except FileNotFoundError:
            counts = None

        for key, values in table.items():
            if key not in accumulated:
                accumulated[key] = [[] for _ in values]
            for i, value in enumerate(values):
                weight = 1 if counts is None else counts.get(f"{key}|{i}", 0)
                accumulated[key][i].append((value, weight))

    merged = {}
    for key, per_action in accumulated.items():
        row = []
        for entries in per_action:
            total_weight = sum(weight for _, weight in entries)
            if total_weight > 0:
                row.append(
                    sum(value * weight for value, weight in entries) / total_weight
                )
            else:
                row.append(entries[0][0])
        merged[key] = row

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
    global _CURRENT_BUDGET

    worker_dir = WORKER_DIR / gif_prefix
    worker_dir.mkdir(parents=True, exist_ok=True)

    start_round, epsilon, total_episodes, successes, best_demo_key = initial_state(
        state_path, model_path
    )

    signal.signal(signal.SIGINT, _request_graceful_stop)
    print(
        f"{num_workers} workers, {episodes_per_round} episodes per round. "
        f"Ctrl-C finishes the current round and saves it before exiting."
    )

    for round_num in range(start_round, max_rounds + 1):
        table_paths = [worker_dir / f"worker{i}.json" for i in range(num_workers)]
        summary_paths = [worker_dir / f"worker{i}_summary.json" for i in range(num_workers)]

        # The round's episode budget, claimed one at a time by whichever
        # worker is free. Created from the spawn context so the shared
        # lock survives being passed to spawned children.
        remaining = _SPAWN_CTX.Value("i", episodes_per_round)
        _CURRENT_BUDGET = remaining

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
        try:
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            # A second Ctrl-C during the drain. Workers ignore SIGINT, so
            # they have to be stopped explicitly, and nothing from this
            # round is trustworthy afterwards -- the last completed round
            # is still safely on disk.
            print("Abandoning this round; the last saved round is unaffected.")
            for process in processes:
                process.terminate()
            for process in processes:
                process.join()
            return

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

        # Skipped when stopping: the demo is an observability nicety, not
        # part of the checkpoint, and someone who just asked for their
        # cores back shouldn't wait a whole extra episode for a GIF. The
        # previous best_demo_key carries forward untouched.
        if not _STOP_REQUESTED:
            demo_env = env_class(max_steps=max_steps)
            demo_agent = QLearningAgent(num_actions=num_actions())
            demo_agent.load(model_path)
            demo = run_demo_episode(demo_env, demo_agent, max_steps)
            demo_env.close()

            save_gif(demo["frames"], f"{gif_prefix}_progress_round{round_num:03d}.gif")
            print(
                f"  [demo] reached_goal={demo['reached_goal']} "
                f"tiles_visited={demo['tiles_visited']} steps={demo['steps']} "
                f"nudges={demo['nudges']}"
            )

            demo_key = (demo["reached_goal"], demo["tiles_visited"], -demo["steps"])
            if best_demo_key is None or demo_key > best_demo_key:
                best_demo_key = demo_key
                save_gif(demo["frames"], f"{gif_prefix}_best_so_far.gif")
                print(f"  [demo] new best so far (round {round_num})")

        save_progress(state_path, round_num, epsilon, total_episodes, successes, best_demo_key)

        if _STOP_REQUESTED:
            print(
                f"Saved through round {round_num}. Relaunching resumes from "
                f"round {round_num + 1} -- with a different worker count if you "
                f"want, since it's read at launch."
            )
            return

    print()
    print(f"Finished {max_rounds} rounds. Total successes: {successes}/{total_episodes}")
