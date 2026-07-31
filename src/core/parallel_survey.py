import os
import pickle
import multiprocessing as mp
from collections import deque

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT
from core.memory import get_player_position
from core.pathfind import _snapshot, _restore, _step, DIRECTIONS
from core.scheduling import apply_worker_qos, decide_yield, mark_decision_for_workers

# Workers are created with the 'spawn' start method explicitly, not this
# platform's 'fork' default. Found the hard way: a real run hung solid
# after 26 healthy rounds, every stuck worker parked in a kernel
# futex_wait -- the signature of forking a process that has already
# initialized a threaded native runtime (PyTorch/OpenMP's own thread
# pool here) elsewhere in memory, a well-documented fork hazard, not a
# bug in this module's own logic. It only bit intermittently because the
# driver process imports stable_baselines3 (to get a *reference* to a
# build_handle_battle function to hand workers) before ever forking, so
# torch gets initialized in the parent regardless of workers never
# calling DQN.load() there themselves. 'spawn' starts each worker as a
# brand new interpreter that imports everything itself, sidestepping the
# hazard entirely -- exactly why build_handle_battle/build_heal_if_needed
# were already required to be plain top-level functions rather than
# closures: spawn needs real pickling, not fork's copy-on-write reuse.
_SPAWN_CTX = mp.get_context("spawn")

# Parallel counterpart to pathfind.survey_map, for this machine's full
# core count rather than one.
#
# survey_map explores one tile at a time in a single PyBoy process, and
# that process spends nearly all its wall-clock time simulating frames
# -- fighting trainers, waiting out walk/battle animations -- not doing
# anything the Python interpreter's GIL would have serialized anyway.
# Running N independent PyBoy instances in N processes is close to a
# true N-times speedup on that time, not just a paper one. This is the
# same reasoning train_route1_agent_parallel.py already relies on for
# training; this module applies it to exploration instead.
#
# Round-based, mirroring that same script's pattern: each round, the
# current frontier (tiles discovered but not yet explored) is split
# across a worker pool. Every worker restores each of its assigned
# tiles' saved states in turn and tries all four directions from it --
# fighting/healing exactly as the single-process version does -- then
# reports back whatever new tiles and exits it found. The driver merges
# all of that (deduplicating a tile two different workers, or a worker
# and an already-known tile, both happened to reach independently) into
# the shared frontier for the next round, and repeats until nothing is
# left to explore or the tile cap is hit.
#
# Workers are spawned fresh each round (not a long-lived pool) rather
# than sharing anything already loaded in the driver -- matching
# train_route1_agent_parallel.py's own workers, which each independently
# build their own env/agent inside the child process. That costs a
# little redundant model-loading time every round, but each round's
# actual workload (potentially dozens of trainer battles and heal round
# trips per worker) dwarfs it, and it sidesteps any risk of forking a
# process that has already loaded a PyTorch model into an inconsistent
# state -- untested territory in this codebase, unlike the plain
# JSON-table Q-learning workers that pattern was proven on.

# One worker per core. This was written on a 4-core container, and a
# hardcoded 4 left most of the CPU idle anywhere with more -- the whole
# point of the parallel engine is to use what's actually available.
# POKEMON_RED_WORKERS overrides it, for deliberately running smaller
# (leaving cores free for something else, or reproducing a specific run).
#
# Worth knowing before raising this a lot: each worker is a full PyBoy
# plus, for any map with trainers, its own copy of the trainer-battle
# DQN, which measured ~700MB resident. 18 workers is therefore ~12.6GB,
# so on a high-core machine memory, not cores, is usually the real cap.
NUM_WORKERS = int(os.environ.get("POKEMON_RED_WORKERS") or (os.cpu_count() or 4))
TILES_PER_WORKER_PER_ROUND = 20

# PyBoy is the bottleneck here and it is single-threaded C, so each
# worker wants exactly one compute thread. Torch otherwise defaults to
# one thread per core *per process*: at 18 workers on 18 cores that is
# 324 threads contending for 18 cores, which is slower than not
# threading the model at all. Set before workers spawn so they inherit
# it, since torch reads this at import time and the children re-import
# everything from scratch under 'spawn'.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _run_worker(assigned, start_map, build_handle_battle, build_heal_if_needed,
                 output_path, capture_frames):
    """
    assigned: list of ((x, y), snapshot_bytes) this worker explores.

    build_handle_battle/build_heal_if_needed: zero-arg top-level
    functions (must be importable by reference, not local closures --
    picklable regardless of start method) each worker calls once to
    build its own handle_battle/heal_if_needed, e.g. by loading its own
    copy of a trained model. `build_heal_if_needed` receives the
    handle_battle just built.
    """
    apply_worker_qos()
    pyboy = create_emulator()

    handle_battle = build_handle_battle() if build_handle_battle else None
    heal_if_needed = build_heal_if_needed(handle_battle) if build_heal_if_needed else None

    new_tiles = {}      # (x, y) -> snapshot bytes, for tiles not seen before this round
    exits = {}          # ((x, y), direction) -> (map_id, x, y)
    refreshed = {}      # (x, y) -> snapshot bytes, for an assigned tile healed to a new state
    edges = {}          # ((x, y), direction) -> (x, y), every successful move tried this
                         # round, not just ones landing on a first-discovered tile -- the
                         # real walkable-adjacency graph, including any one-way edge (a
                         # ledge can be walked down but not back up) a same-tile-distance
                         # geometric guess would get wrong.
    frames = [] if capture_frames else None
    errors = []  # (key, direction, repr(exception)), for a battle handler that raises
    retry_keys = []  # keys whose heal step raised -- see below for why these need
                      # to go back in the driver's queue rather than just being logged

    for key, snapshot in assigned:
        _restore(pyboy, snapshot)

        try:
            if heal_if_needed is not None and heal_if_needed(pyboy, key):
                snapshot = _snapshot(pyboy)
                refreshed[key] = snapshot
        except Exception as exc:
            # heal_if_needed already treats its own travel failures as
            # best-effort (returns False rather than raising), but a
            # handle_battle passed through it can still raise -- a
            # not-yet-discovered trainer encountered mid-heal-trip, say.
            #
            # `continue` here skips this key's whole direction loop for
            # the round -- and a key is only ever popped from the
            # driver's queue once, so on its own that silently abandons
            # this tile's exploration forever, not just for this round.
            # That's exactly what happened live: Route 3's survey came
            # back "complete" with its frontier dead-ending right at the
            # three tiles whose heal step is logged raising here, one
            # of which sits on a real warp (a same-map-ID teleport this
            # project hadn't seen before) that a next round might well
            # have walked straight through. retry_keys is how the driver
            # gets a chance to try these tiles again instead of treating
            # one bad-luck battle as if the tile had genuinely explored
            # to a dead end.
            errors.append((key, "heal", repr(exc)))
            retry_keys.append(key)
            continue

        for direction in DIRECTIONS:
            _restore(pyboy, snapshot)
            try:
                moved = _step(pyboy, direction, handle_battle=handle_battle)
            except Exception as exc:
                # A raised handle_battle (survey_viridian_forest's aborts
                # loudly on a lost trainer fight, by design -- the battle
                # policy is not 100% deterministic, so a loss during a
                # long survey is a real, if rare, possibility rather than
                # a bug). Before this, an uncaught exception here killed
                # the whole worker process without writing its output
                # file, and the driver's own open() on the now-missing
                # path then raised an unrelated FileNotFoundError --
                # burying the actual cause behind a confusing symptom,
                # exactly what happened live surveying Route 3. Recording
                # it here and moving to the next direction costs one
                # untried move, not the whole worker's remaining tiles.
                errors.append((key, direction, repr(exc)))
                continue
            position = get_player_position(pyboy)

            if position["map_id"] != start_map:
                exits[(key, direction)] = (position["map_id"], position["x"], position["y"])
                continue
            if not moved:
                continue

            next_key = (position["x"], position["y"])
            edges[(key, direction)] = next_key

            if next_key in new_tiles:
                continue

            new_snapshot = _snapshot(pyboy)
            new_tiles[next_key] = new_snapshot
            if capture_frames:
                frames.append((position["x"], position["y"], pyboy.screen.image.convert("RGB")))

    pyboy.stop()

    with open(output_path, "wb") as f:
        pickle.dump(
            {"new_tiles": new_tiles, "exits": exits, "refreshed": refreshed,
             "edges": edges, "frames": frames, "errors": errors,
             "retry_keys": retry_keys},
            f,
        )


def parallel_survey_map(save_state_path, max_tiles=5000, build_handle_battle=None,
                         build_heal_if_needed=None, capture_frames=False,
                         num_workers=NUM_WORKERS, worker_dir=None, progress=True):
    """
    Parallel counterpart to pathfind.survey_map: same exhaustive,
    save-state-restoring flood fill of the map the player starts on, but
    splitting each round's frontier across `num_workers` independent
    PyBoy processes instead of walking it one tile at a time in this one.

    Starts fresh from `save_state_path` rather than an already-running
    emulator -- unlike the single-process version, there is no live
    session to hand off partway through, since each worker needs its own
    separate PyBoy instance.

    Returns (tiles, exits, complete, frames, start_map, states, edges),
    matching survey_map's own (tiles, exits, complete) plus the captured
    frames (only populated if capture_frames=True), the map ID surveyed
    (so a caller doesn't need to reload the state just to learn it), the
    full key -> snapshot-bytes map -- letting a caller jump straight to
    any discovered tile, including one just past an exit, by restoring
    its snapshot and stepping across, rather than re-searching for it --
    and the real walkable-adjacency graph as ((x, y), direction) -> (x,
    y), every successful move actually tried during the survey rather
    than just the spanning-tree edges used for first discovery. That
    makes it exact (a one-way ledge shows up as present in one direction
    and absent in the other) where guessing adjacency from tile geometry
    alone would not, which matters for anything computing shortest-path
    distances over the map, e.g. reward shaping for a navigation agent.
    """
    # Same core-tier courtesy as training: a deliberately partial worker
    # count means the user is keeping cores, and the kept cores should be
    # the best ones. Decided here, applied by each worker to itself.
    mark_decision_for_workers(decide_yield(num_workers))


    worker_dir = worker_dir or (PROJECT_ROOT / "models" / "parallel_survey_workers")
    worker_dir.mkdir(parents=True, exist_ok=True)

    pyboy = create_emulator()
    load_state(pyboy, save_state_path)
    run_frames(pyboy, 30)

    start = get_player_position(pyboy)
    start_map = start["map_id"]
    start_key = (start["x"], start["y"])
    origin_snapshot = _snapshot(pyboy)

    frames = []
    if capture_frames:
        frames.append((start["x"], start["y"], pyboy.screen.image.convert("RGB")))
    pyboy.stop()

    tiles = {start_key}
    states = {start_key: origin_snapshot}
    exits = {}
    edges = {}
    queue = deque([start_key])
    # A key whose heal step raised gets requeued (see _run_worker) rather
    # than silently abandoned -- but bounded, so a tile that is genuinely,
    # persistently unlucky (or sits next to a trainer the battle policy
    # keeps losing to) can't loop forever and block the survey from ever
    # reporting complete.
    retry_counts = {}
    MAX_RETRIES = 3

    round_num = 0
    while queue and len(tiles) < max_tiles:
        round_num += 1

        batch = []
        while queue and len(batch) < num_workers * TILES_PER_WORKER_PER_ROUND:
            key = queue.popleft()
            batch.append((key, states[key]))

        chunks = [batch[i::num_workers] for i in range(num_workers)]
        chunks = [c for c in chunks if c]

        output_paths = [worker_dir / f"round{round_num}_worker{i}.pkl" for i in range(len(chunks))]
        processes = [
            _SPAWN_CTX.Process(
                target=_run_worker,
                args=(chunks[i], start_map, build_handle_battle, build_heal_if_needed,
                      output_paths[i], capture_frames),
            )
            for i in range(len(chunks))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        new_this_round = 0
        for worker_index, path in enumerate(output_paths):
            try:
                with open(path, "rb") as f:
                    result = pickle.load(f)
            except FileNotFoundError:
                # _run_worker now catches everything survey-related
                # itself (a lost trainer battle, a failed heal trip) and
                # keeps going -- this path is left for what it can't
                # catch, like the process dying outright. Previously an
                # uncaught exception anywhere in a worker crashed the
                # whole build() call on a FileNotFoundError that named
                # the missing pickle, not the real cause; requeuing here
                # at least keeps the survey itself alive and correct
                # (its assigned tiles are retried, not silently dropped
                # from the count as if they had been explored) even
                # though the underlying crash still deserves a look.
                lost = chunks[worker_index]
                print(f"  worker {worker_index} produced no output "
                      f"(process crashed); requeueing its {len(lost)} tiles")
                for key, _ in lost:
                    queue.append(key)
                continue
            path.unlink()

            if result.get("errors"):
                for key, direction, error in result["errors"]:
                    print(f"  worker {worker_index}: {key} {direction} -> {error}")

            for key in result.get("retry_keys", []):
                retry_counts[key] = retry_counts.get(key, 0) + 1
                if retry_counts[key] <= MAX_RETRIES:
                    queue.append(key)
                else:
                    print(f"  {key}: heal step failed {retry_counts[key]} times, "
                          f"giving up on exploring past it")

            for key, snap in result["refreshed"].items():
                states[key] = snap

            for key, snap in result["new_tiles"].items():
                if key in tiles:
                    continue
                tiles.add(key)
                states[key] = snap
                queue.append(key)
                new_this_round += 1

            exits.update(result["exits"])
            edges.update(result["edges"])

            if capture_frames and result["frames"]:
                frames.extend(result["frames"])

        if progress:
            print(
                f"round {round_num}: {len(chunks)} workers, {len(batch)} tiles explored, "
                f"{new_this_round} new ({len(tiles)} total, {len(queue)} pending)"
            )

    complete = not queue
    return tiles, exits, complete, frames, start_map, states, edges
