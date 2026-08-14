import json
from collections import deque

from core.config import SCREENSHOT_DIR

ROUTE_3_MAP_ID = 14
MT_MOON_MAP_ID = 15
ROUTE_3_META_PATH = SCREENSHOT_DIR / "route3_map_meta.json"

# GOAL_TILE used to be (22, 10), the deepest point an "exhaustive" survey
# could reach, on the theory (wrong, see below) that Mt. Moon was blocked
# by something this project had no way past yet. It wasn't -- that
# survey had a real blind spot, and the true exit is here: five tiles
# at (57-61, 0), each one "up" away from Mt. Moon itself (map 15). Any
# of them works identically; (59, 0) is just the middle one, picked the
# same way the forest env picks a single representative tile for its
# own goal.
GOAL_TILE = (59, 0)

# Why the first survey missed 300+ tiles and the real exit entirely:
# Gen 1 trainers only ever battle you once. The survey (and every
# automated re-check before this) always restored to a pre-battle
# snapshot before testing each of a tile's four directions, so it could
# only ever see the world as it looks with every nearby trainer still
# undefeated -- some of whom sit directly in the way, sightline
# triggering a battle-and-relocate the instant you approach, exactly
# the "impossible edge" pattern already documented below. What it could
# never see is what's past them *after* they're beaten: once defeated,
# a trainer's sightline never triggers again, and walking through
# resumes normally rather than intercepting into another fight.
#
# Found by a live playthrough of the exact button sequence past what
# every automated check called a wall (a human, not the survey,
# noticing the trainer "sees you, walks up, and battles" rather than
# treating that as terrain) -- confirmed by hand-verifying identical
# tiles with a fresh re-exploration where the local trainers were
# already beaten: (11,6)/(14,6)/(19,6)'s recorded moves flip from
# relocate-on-battle to a plain adjacent step once their trainer is no
# longer undefeated. Those flips are honored here (the newer,
# post-defeat edges win when the two surveys disagree) since a trained
# policy will have beaten every trainer along its own path long before
# needing to walk back through the same tile.
def _load_distances():
    """
    Shortest-path distance to GOAL_TILE for every Route 3 tile, using
    the real walkable-adjacency graph two merged surveys recorded
    (edges: ((x, y), direction) -> (x, y)) -- see the module docstring
    above for why a geometry-based guess would get this map especially
    wrong, on top of the reasons the forest env's own graph-based
    shaping already covers (ledges, long trainer-relocation jumps).

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
    the signature of a trainer battle relocating the player rather than
    a plain step (see the module docstring). These are exactly the
    tiles worth paying _try_engage_trainer's probe cost from, the same
    role forest_env.py's KNOWN_TRAINER_TILES plays there, just derived
    from this survey's own graph instead of separate per-trainer capture
    files. Being over-inclusive here is harmless -- a small number of
    these are probably genuine one-way ledges rather than trainers, and
    probing a ledge tile just costs one wasted button-press attempt, not
    a correctness bug.
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
    corridor, on top of which several edges are the trainer-relocation
    jumps documented above.

    Mt. Moon itself gets potential 0, matching GOAL_TILE's own distance
    -- continuity across the boundary, so stepping through the exit is
    never scored as a large penalty relative to standing right next to
    it. Anywhere else off Route 3 (the 4 known exits back to Pewter)
    gets the worst-case anchor, the same idea as every other navigation
    env's off-route case.
    """
    if position["map_id"] == MT_MOON_MAP_ID:
        return 0
    if position["map_id"] != ROUTE_3_MAP_ID:
        return -_MAX_DISTANCE

    tile = (position["x"], position["y"])
    return -_DISTANCES.get(tile, _MAX_DISTANCE)


def calculate_route3_reward(before, after):
    """
    Same shape as the forest reward: a small step cost, a bigger cost for
    not moving at all, potential-based shaping toward the goal, a large
    terminal reward for reaching it, and a penalty for leaving Route 3
    any other way -- which also covers losing a forced trainer battle,
    since a blackout lands the player on Pewter's Pokemon Center map, not
    Route 3 or Mt. Moon.
    """
    reward = -0.01

    if position_key(before) == position_key(after):
        reward -= 0.25

    reward += route3_potential(after) - route3_potential(before)

    if after["map_id"] == MT_MOON_MAP_ID:
        reward += 100.0
    elif after["map_id"] != ROUTE_3_MAP_ID:
        reward -= 20.0

    return reward
