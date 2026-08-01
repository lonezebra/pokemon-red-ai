import json
from collections import deque

from core.config import SCREENSHOT_DIR

ROUTE_3_MAP_ID = 14
ROUTE_3_META_PATH = SCREENSHOT_DIR / "route3_map_meta.json"

# The deepest point an exhaustive parallel survey could reach (its
# frontier fully exhausted -- 125/125 tiles, 0 new in the final round),
# not an exit: every recorded exit from Route 3 leads back to Pewter, and
# the only way further east is blocked by what looks like a Cut tree /
# boulder line this project has no way to clear yet (no HM01, which
# normally requires either an Oak's Aide reward or reaching the S.S.
# Anne -- both well beyond where the project currently stands). Reusing
# this survey's graph rather than re-deriving it live is deliberate, the
# same reasoning as the forest env: some of this route's edges are not
# plain 1-tile steps (see below), so a geometry-based guess would get
# several of them wrong.
#
# Found by forward BFS from the entrance (0, 9) over the same graph
# _load_distances() below walks backward: (22, 10) is the unique tile at
# the maximum distance, 30 hops.
GOAL_TILE = (22, 10)


def _load_distances():
    """
    Shortest-path distance to GOAL_TILE for every Route 3 tile, using the
    real walkable-adjacency graph a parallel survey recorded (edges:
    ((x, y), direction) -> (x, y)) rather than one guessed from tile
    geometry -- see rewards/forest_rewards.py for why that distinction
    matters in general.

    It matters especially here: several of this route's trainers have a
    long sightline and relocate the player to their own fixed approach
    tile on a win, which can be many tiles from wherever the fight
    actually triggered -- e.g. (14,9) pressing "down" resolves a battle
    and lands back on (14,9) itself, and (11,6) resolves one and lands
    near (10,4)/(12,4)/(11,5) depending on which direction triggered it.
    Confirmed live, not guessed: reproduced several of these exactly by
    replaying the same battle handler the survey used. None of that
    breaks the graph as a graph -- every edge is still a real,
    deterministic, winnable transition -- it just means "distance" here
    is graph hops, not tile-geometry distance, exactly the ledge case
    the forest reward already handles the same way.

    Standard reverse BFS: walks the graph backward from the goal along
    every edge's reverse, so a node reached after N reversed hops is
    exactly N real forward hops from the goal.
    """
    with open(ROUTE_3_META_PATH) as f:
        meta = json.load(f)

    reverse_adjacency = {}
    for edge in meta["edges"]:
        frm = tuple(edge["from"])
        to = tuple(edge["to"])
        reverse_adjacency.setdefault(to, []).append(frm)

    distances = {GOAL_TILE: 0}
    queue = deque([GOAL_TILE])
    while queue:
        node = queue.popleft()
        for neighbor in reverse_adjacency.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return distances


def _load_trainer_trigger_tiles():
    """
    Tiles whose recorded edges include at least one non-adjacent jump --
    the signature of a trainer battle relocating the player rather than a
    plain step (see _load_distances' docstring). These are exactly the
    tiles worth paying _try_engage_trainer's probe cost from, the same
    role forest_env.py's KNOWN_TRAINER_TILES plays there, just derived
    from this survey's own graph instead of separate per-trainer capture
    files (Route 3's survey never captured those individually). Being
    over-inclusive here is harmless -- a small number of these are
    probably genuine one-way ledges rather than trainers, and probing a
    ledge tile just costs one wasted button-press attempt, not a
    correctness bug.
    """
    with open(ROUTE_3_META_PATH) as f:
        meta = json.load(f)

    deltas = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
    tiles = set()
    for edge in meta["edges"]:
        fx, fy = edge["from"]
        tx, ty = edge["to"]
        dx, dy = deltas[edge["direction"]]
        if (fx + dx, fy + dy) != (tx, ty):
            tiles.add((fx, fy))
    return tiles


_DISTANCES = _load_distances()
_MAX_DISTANCE = max(_DISTANCES.values())
KNOWN_TRAINER_TILES = _load_trainer_trigger_tiles()


def position_key(position):
    return (position["map_id"], position["x"], position["y"])


def route3_potential(position):
    """
    Higher = closer to the goal. Same potential-based-shaping pattern as
    the forest env -- distance here is graph shortest-path rather than a
    raw -y coordinate, for the same reason: this route is not a straight
    corridor, on top of which several edges are not plain 1-tile steps.

    Anywhere off Route 3 (the 4 known exits back to Pewter) gets the
    worst-case anchor, the same idea as every other navigation env's
    off-route case.
    """
    if position["map_id"] != ROUTE_3_MAP_ID:
        return -_MAX_DISTANCE

    tile = (position["x"], position["y"])
    return -_DISTANCES.get(tile, _MAX_DISTANCE)


def calculate_route3_reward(before, after):
    """
    Same shape as the forest reward: a small step cost, a bigger cost for
    not moving at all, potential-based shaping toward the goal, a large
    terminal reward for reaching it, and a penalty for leaving Route 3
    anywhere else -- which also covers losing a forced trainer battle,
    since a blackout lands the player on Pewter's Pokemon Center map, not
    Route 3.

    Unlike the forest (whose goal is stepping through an exit onto a
    different map), reaching GOAL_TILE here is a same-map event -- there
    is no further map to cross into, since it is the deepest point this
    project can currently reach, not a real exit.
    """
    reward = -0.01

    if position_key(before) == position_key(after):
        reward -= 0.25

    reward += route3_potential(after) - route3_potential(before)

    reached_goal = (
        after["map_id"] == ROUTE_3_MAP_ID
        and (after["x"], after["y"]) == GOAL_TILE
    )
    if reached_goal:
        reward += 100.0
    elif after["map_id"] != ROUTE_3_MAP_ID:
        reward -= 20.0

    return reward
