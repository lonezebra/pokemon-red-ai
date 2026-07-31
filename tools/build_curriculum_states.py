"""
Capture Viridian Forest save states at fixed shortest-path distances
from the goal, for curriculum training.

    .venv/bin/python3 tools/build_curriculum_states.py [--every N]

Writes saves/forest_curriculum/d<NNN>_<x>_<y>.state, one per checkpoint.

Why this exists: training the forest from its entrance stopped working.
A run restored to a policy that greedily walked 106 tiles to within 33
hops of the goal degraded to 37 tiles / 92 hops within 400 episodes,
with the reward function verified clean (no value inflation, maxQ 97.8
against a 101 ceiling) and the repair verified harmless (1 value
clamped, 0 argmax ties). From the entrance, a success is a 127-move
correct sequence; at epsilon 0.14 essentially every episode ends in
failure, and backward replay then propagates full-strength failure
signal along whole trajectories far more often than it propagates a
success. Starting episodes near the goal inverts that ratio -- 10 moves
from the exit, a mostly-greedy policy succeeds often -- so the majority
signal being propagated is the one worth learning.

How the walk works: rather than replaying a precomputed move list, this
re-derives the shortest path from wherever the player actually is on
every iteration and takes only its first move. Divergence is expected
and self-correcting that way -- a forced trainer battle, a failed step,
or the emulator settling on an unexpected tile just changes the next
lookup instead of desynchronizing a fixed script.

The path follows the same real walkable-adjacency graph the reward
function's distances come from (survey-recorded traversals, including
one-way ledges), so a checkpoint at distance d is genuinely d moves
from the exit rather than d tiles away in straight-line terms.
"""

import argparse
import json
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from stable_baselines3 import DQN

from core.battle_runner import fight_current_battle
from core.config import PROJECT_ROOT, SCREENSHOT_DIR
from core.controls import wait_for_free_movement
from core.emulator import create_emulator, run_frames
from core.memory import get_player_position
from core.pathfind import _step
from core.state import load_state, save_state
from rewards.forest_rewards import _DISTANCES, FOREST_MAP_ID, GOAL_TILE

ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "leveled.state"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
OUTPUT_DIR = PROJECT_ROOT / "saves" / "forest_curriculum"

# Generous: the shortest path is 127 moves, and trainer fights plus the
# occasional blocked step mean the real walk is longer than the ideal one.
MAX_MOVES = 600


def load_forward_graph():
    """((x, y), direction) -> (x, y) as directed adjacency, one entry per
    direction the survey actually walked successfully."""
    with open(SCREENSHOT_DIR / "forest_map_meta.json") as f:
        meta = json.load(f)

    graph = {}
    for edge in meta["edges"]:
        graph.setdefault(tuple(edge["from"]), []).append(
            (edge["direction"], tuple(edge["to"]))
        )
    return graph


def next_move_toward_goal(graph, tile, target=GOAL_TILE):
    """
    First move of a shortest path from `tile` to `target` (the goal by
    default), or None if it is unreachable from here.

    Forward BFS rather than a greedy descent of the precomputed
    distance-to-goal map: those distances are correct, but reading a move
    off them still requires knowing which neighbour is one hop closer,
    and one-way edges mean "adjacent tile with distance d-1" is not
    always reachable from here. Searching forward over the real edges
    answers the question directly.
    """
    if tile == target:
        return None

    previous = {tile: None}
    queue = deque([tile])
    found = False
    while queue:
        node = queue.popleft()
        if node == target:
            found = True
            break
        for direction, neighbor in graph.get(node, []):
            if neighbor not in previous:
                previous[neighbor] = (node, direction)
                queue.append(neighbor)

    if not found:
        return None

    node = target
    while previous[node] is not None:
        parent, direction = previous[node]
        if parent == tile:
            return direction
        node = parent
    return None


def checkpoint_distances(every):
    """Every `every` hops from the goal, deepest first, excluding the
    entrance itself -- an episode starting there is the from-scratch case
    the curriculum exists to avoid."""
    start_distance = _DISTANCES.get((17, 47))
    return [d for d in range(every, start_distance, every)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=int, default=10,
                        help="spacing between checkpoints in shortest-path hops")
    args = parser.parse_args()

    wanted = set(checkpoint_distances(args.every))
    if not wanted:
        print("no checkpoints requested")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    graph = load_forward_graph()

    pyboy = create_emulator()
    load_state(pyboy, ENTRY_STATE_PATH)
    run_frames(pyboy, 30)

    trainer_model = DQN.load(str(TRAINER_MODEL_PATH))

    def handle_battle(emulator):
        fight_current_battle(emulator, trainer_model)
        wait_for_free_movement(emulator)

    captured = {}
    for move_number in range(MAX_MOVES):
        position = get_player_position(pyboy)
        if position["map_id"] != FOREST_MAP_ID:
            print(f"left the forest at move {move_number} (map "
                  f"{position['map_id']}) -- stopping")
            break

        tile = (position["x"], position["y"])
        distance = _DISTANCES.get(tile)

        # Capture on arrival, before moving on. Any checkpoint the walk
        # passes through gets taken the first time it is stood on, so a
        # detour that revisits it later cannot overwrite the state with a
        # worse-off one (lower HP after a fight, say).
        if distance in wanted and distance not in captured:
            path = OUTPUT_DIR / f"d{distance:03d}_{tile[0]}_{tile[1]}.state"
            save_state(pyboy, path)
            captured[distance] = path
            print(f"  d={distance:3d} at {tile}: {path.name}")

        if tile == GOAL_TILE or not wanted - set(captured):
            break

        direction = next_move_toward_goal(graph, tile)
        if direction is None:
            print(f"no route to the goal from {tile} -- stopping")
            break

        _step(pyboy, direction, handle_battle=handle_battle)

    pyboy.stop()

    missing = sorted(wanted - set(captured))
    print(f"\ncaptured {len(captured)}/{len(wanted)} checkpoints in {OUTPUT_DIR}")
    if missing:
        print(f"missing distances: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
