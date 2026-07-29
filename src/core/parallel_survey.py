import pickle
import multiprocessing as mp
from collections import deque

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT
from core.memory import get_player_position
from core.pathfind import _snapshot, _restore, _step, DIRECTIONS

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

NUM_WORKERS = 4
TILES_PER_WORKER_PER_ROUND = 20


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
    pyboy = create_emulator()

    handle_battle = build_handle_battle() if build_handle_battle else None
    heal_if_needed = build_heal_if_needed(handle_battle) if build_heal_if_needed else None

    new_tiles = {}      # (x, y) -> snapshot bytes, for tiles not seen before this round
    exits = {}          # ((x, y), direction) -> (map_id, x, y)
    refreshed = {}      # (x, y) -> snapshot bytes, for an assigned tile healed to a new state
    frames = [] if capture_frames else None

    for key, snapshot in assigned:
        _restore(pyboy, snapshot)

        if heal_if_needed is not None and heal_if_needed(pyboy, key):
            snapshot = _snapshot(pyboy)
            refreshed[key] = snapshot

        for direction in DIRECTIONS:
            _restore(pyboy, snapshot)
            moved = _step(pyboy, direction, handle_battle=handle_battle)
            position = get_player_position(pyboy)

            if position["map_id"] != start_map:
                exits[(key, direction)] = (position["map_id"], position["x"], position["y"])
                continue
            if not moved:
                continue

            next_key = (position["x"], position["y"])
            if next_key in new_tiles:
                continue

            new_snapshot = _snapshot(pyboy)
            new_tiles[next_key] = new_snapshot
            if capture_frames:
                frames.append((position["x"], position["y"], pyboy.screen.image.convert("RGB")))

    pyboy.stop()

    with open(output_path, "wb") as f:
        pickle.dump(
            {"new_tiles": new_tiles, "exits": exits, "refreshed": refreshed, "frames": frames},
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

    Returns (tiles, exits, complete, frames, start_map, states), matching
    survey_map's own (tiles, exits, complete) plus the captured frames
    (only populated if capture_frames=True), the map ID surveyed (so a
    caller doesn't need to reload the state just to learn it), and the
    full key -> snapshot-bytes map -- letting a caller jump straight to
    any discovered tile, including one just past an exit, by restoring
    its snapshot and stepping across, rather than re-searching for it.
    """

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
    queue = deque([start_key])

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
            mp.Process(
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
        for path in output_paths:
            with open(path, "rb") as f:
                result = pickle.load(f)
            path.unlink()

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

            if capture_frames and result["frames"]:
                frames.extend(result["frames"])

        if progress:
            print(
                f"round {round_num}: {len(chunks)} workers, {len(batch)} tiles explored, "
                f"{new_this_round} new ({len(tiles)} total, {len(queue)} pending)"
            )

    complete = not queue
    return tiles, exits, complete, frames, start_map, states
