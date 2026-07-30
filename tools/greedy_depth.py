"""
Greedy-walk analysis of a forest Q-table: how deep does the policy get?

    .venv/bin/python3 tools/greedy_depth.py <q_table.json>

Simulates the pure-greedy walk from the forest entrance against the
survey's recorded edge graph (real traversals, including one-way ledges)
and reports the deepest point reached in shortest-path hops, plus where
and why it stops -- a cycle's tiles and their Q-rows, or the frontier
tile with no entry yet.

This is the offline readout behind the demo line's depth=N: it needs no
emulator, runs in milliseconds, and can be pointed at any table -- e.g.
one fetched from origin mid-run (git show origin/<branch>:models/
forest_q_table.json > /tmp/q.json) to watch a training run remotely.

Exit code 0 with depth 0 means the table solves the maze greedily.
"""

import json
import pathlib
import sys
from collections import deque

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
META_PATH = PROJECT_ROOT / "screenshots" / "forest_map_meta.json"

DIRS = {0: "up", 1: "down", 2: "left", 3: "right"}
START = (17, 47)
GOAL = (1, 1)


def load_graph():
    meta = json.load(open(META_PATH))
    edges = {}
    reverse = {}
    for e in meta["edges"]:
        frm, to = tuple(e["from"]), tuple(e["to"])
        edges[(frm, e["direction"])] = to
        reverse.setdefault(to, []).append(frm)
    dist = {GOAL: 0}
    queue = deque([GOAL])
    while queue:
        node = queue.popleft()
        for nb in reverse.get(node, []):
            if nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return edges, dist


def analyze(q, edges, dist, max_steps=600):
    tile = START
    seen_pairs = {}
    trajectory = []
    stop = None
    for step in range(max_steps):
        row = q.get(f"51,{tile[0]},{tile[1]}")
        if row is None:
            stop = ("frontier", tile)
            break
        best = max(range(4), key=lambda i: row[i])
        if (tile, best) in seen_pairs:
            stop = ("cycle", trajectory[seen_pairs[(tile, best)]:])
            break
        seen_pairs[(tile, best)] = step
        nxt = edges.get((tile, DIRS[best]), tile)
        trajectory.append((tile, DIRS[best], row[best], nxt))
        if dist.get(nxt) == 0 or nxt == GOAL:
            trajectory.append((nxt, "-", 0.0, nxt))
            stop = ("goal", nxt)
            break
        tile = nxt

    tiles = [t for t, _, _, _ in trajectory]
    depth = min((dist.get(t, 9999) for t in tiles), default=9999)
    return depth, stop, trajectory


def main():
    q_path = sys.argv[1] if len(sys.argv) > 1 else str(
        PROJECT_ROOT / "models" / "forest_q_table.json"
    )
    q = json.load(open(q_path))
    edges, dist = load_graph()
    depth, stop, trajectory = analyze(q, edges, dist)

    kind = stop[0] if stop else "step-limit"
    print(f"depth={depth} of {dist[START]}  stop={kind}  "
          f"tiles_walked={len(set(t for t, _, _, _ in trajectory))}")

    if kind == "goal":
        print("GREEDY SOLVES THE MAZE")
        return 0
    if kind == "cycle":
        cyc = stop[1]
        uniq = sorted(set(t for t, _, _, _ in cyc))
        print(f"cycle over {len(uniq)} tiles at dist "
              f"{sorted(set(dist.get(t) for t in uniq))}:")
        for t in uniq[:4]:
            row = q[f"51,{t[0]},{t[1]}"]
            parts = []
            for i, dn in DIRS.items():
                dest = edges.get((t, dn))
                tag = "wall" if dest is None else f"d{dist.get(dest)}"
                parts.append(f"{dn}={row[i]:.2f}({tag})")
            print(f"  {t} d{dist.get(t)}: " + " ".join(parts))
    elif kind == "frontier":
        print(f"greedy walks off its own table at {stop[1]} "
              f"(d{dist.get(stop[1])}) -- no Q entry there yet")
    return 1


if __name__ == "__main__":
    sys.exit(main())
