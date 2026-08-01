"""
What fraction of the forest's tiles does the policy choose correctly?

    .venv/bin/python3 tools/policy_accuracy.py <q_table.json> [...]

Why this exists alongside greedy_depth.py: depth is a *threshold* on
tiny value differences, so it moves in large jumps for reasons that have
nothing to do with the policy getting better or worse. Observed stall
points have hinged on margins of 0.16, 0.18, and 0.04 -- at a 0.04
margin, an update far away that nudges one action by a rounding error
relocates the stall and swings reported depth by tens of hops. Round 66
read depth 36 and round 67 read 68 while the success rate went 18.5% ->
24.5%, which is the signature of a metric dominated by its own
discontinuity rather than by what it is supposed to measure.

Accuracy is continuous instead: every tile is scored independently, so
one flipped near-tie moves the number by one tile's worth rather than
truncating the whole walk. It also measures the entire learned map
rather than only the prefix a greedy walk survives long enough to
reach -- a policy can be improving everywhere beyond its stall point
and depth will not show it.

Two figures are reported:

  overall   -- across every tile with a known distance to the goal.
  on-path   -- across the tiles of one shortest path from the entrance,
               which is what a successful episode actually has to walk.

A tile counts as correct when its greedy action traverses a real edge
to a tile strictly closer to the goal. Actions into walls are wrong by
construction: no edge, no progress.
"""

import json
import pathlib
import sys
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from core.config import SCREENSHOT_DIR
from rewards.forest_rewards import _DISTANCES, GOAL_TILE

FOREST_MAP_ID = 51
START = (17, 47)
# Index order of the Q-table's action columns, matching actions.py.
DIRS = {0: "up", 1: "down", 2: "left", 3: "right"}


def load_edges():
    with open(SCREENSHOT_DIR / "forest_map_meta.json") as f:
        meta = json.load(f)
    edges = {}
    for edge in meta["edges"]:
        edges[(tuple(edge["from"]), edge["direction"])] = tuple(edge["to"])
    return edges


def solution_path(edges):
    """Tiles along one shortest path from the entrance to the goal."""
    adjacency = {}
    for (frm, direction), to in edges.items():
        adjacency.setdefault(frm, []).append(to)

    previous = {START: None}
    queue = deque([START])
    while queue:
        node = queue.popleft()
        if node == GOAL_TILE:
            break
        for neighbor in adjacency.get(node, []):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)

    if GOAL_TILE not in previous:
        return []
    path, node = [], GOAL_TILE
    while node is not None:
        path.append(node)
        node = previous[node]
    return path[::-1]


def is_correct(q, edges, tile):
    """True if the greedy action at `tile` moves strictly closer to the
    goal. None if the table has never seen the tile."""
    row = q.get(f"{FOREST_MAP_ID},{tile[0]},{tile[1]}")
    if row is None:
        return None

    distance = _DISTANCES.get(tile)
    if distance is None or distance == 0:
        return None

    best = max(range(len(row)), key=lambda i: row[i])
    destination = edges.get((tile, DIRS[best]))
    if destination is None:
        return False
    next_distance = _DISTANCES.get(destination)
    return next_distance is not None and next_distance < distance


def score(path, q, edges, tiles):
    correct = seen = 0
    for tile in tiles:
        verdict = is_correct(q, edges, tile)
        if verdict is None:
            continue
        seen += 1
        correct += bool(verdict)
    return correct, seen


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    edges = load_edges()
    path = solution_path(edges)
    everywhere = [t for t in _DISTANCES if _DISTANCES[t] > 0]

    print(f"{'table':<28} {'overall':>16} {'on-path':>16}")
    for arg in sys.argv[1:]:
        table_path = pathlib.Path(arg)
        q = json.loads(table_path.read_text())

        correct, seen = score(path, q, edges, everywhere)
        p_correct, p_seen = score(path, q, edges, path)
        overall = f"{correct}/{seen} = {100.0 * correct / seen:.0f}%" if seen else "-"
        on_path = f"{p_correct}/{p_seen} = {100.0 * p_correct / p_seen:.0f}%" if p_seen else "-"
        print(f"{table_path.name:<28} {overall:>16} {on_path:>16}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
