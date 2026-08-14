import json
from collections import deque

from core.config import SCREENSHOT_DIR

FOREST_MAP_ID = 51
FOREST_META_PATH = SCREENSHOT_DIR / "forest_map_meta.json"

# The forest's only exit toward Pewter City, established by an
# exhaustive parallel survey whose frontier was fully exhausted (see
# build_map_panorama.py / core/parallel_survey.py) -- so this really is
# the complete exit list, not just what a walk happened to find. The
# other 8 exits all lead back to Route 2.
GOAL_TILE = (1, 1)
CONNECTOR_MAP_ID = 47  # what stepping off GOAL_TILE actually leads to


def _load_distances():
    """
    Shortest-path distance to the goal for every forest tile, using the
    *real* walkable-adjacency graph a parallel survey recorded (edges:
    ((x, y), direction) -> (x, y), one entry per direction that actually
    worked) rather than one guessed from tile geometry. That distinction
    matters here specifically: this map has a real one-way section (a
    ledge, most likely, found the hard way when heal-and-return trips
    stopped working past a certain point), so a few edges only exist in
    one direction, and a geometry-based guess would get those wrong.

    Standard reverse BFS: distance-to-goal is directional (a one-way
    edge only helps in one of the two directions), so this walks the
    graph backward from the goal -- along every edge's reverse -- rather
    than forward from each tile individually. A node reached after N
    reversed hops from the goal is exactly N real forward hops from the
    goal.
    """
    with open(FOREST_META_PATH) as f:
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

    # The warp strip above the goal tile. Stepping up from (1,1) exits to
    # map 47, but the game walks the player through (1,0) first, and the
    # env can observe that in-between frame as a forest-map position. The
    # survey never records (1,0) -- it isn't a standable tile -- so
    # without this entry it fell through to the worst-case anchor, making
    # the two halves of a *successful* exit score -147 then +248: a
    # seesaw that inflated goal-side Q-values past the legitimate ~101
    # ceiling (190 observed live) and, once the inflation propagated
    # backward through enough merges, collapsed the training success rate
    # from 27% to 3% in a single round. Zero hops, same as the connector
    # room it is halfway into: crossing (1,1) -> (1,0) -> map 47 then
    # scores +1, 0, +100 -- one clean finish, no seesaw. The other maps'
    # warp strips need no such entry: they lead to worst-case-anchored
    # maps anyway, so their split steps sum to the same total as a direct
    # exit.
    distances[(1, 0)] = 0

    return distances


_DISTANCES = _load_distances()
_MAX_DISTANCE = max(_DISTANCES.values())


def position_key(position):
    return (position["map_id"], position["x"], position["y"])


def forest_potential(position):
    """
    Higher = closer to the goal. Same potential-based-shaping pattern as
    Route 1/Route 2 (see rewards/route2_rewards.py) -- a bigger cost for
    not moving, a small step cost, and shaping toward the goal rather
    than a per-episode "new tile" bonus -- except distance here is graph
    shortest-path rather than a raw -y coordinate. Route 1 and Route 2
    are corridors, where straight-line and shortest-path distance always
    agree; Viridian Forest fills only ~45% of its own bounding box, so
    reaching the goal can genuinely require moving further from it in
    straight-line terms first, to get around a wall -- exactly the case
    a -y-style potential would shape *against* rather than toward.

    The connector room (map 47) gets potential 0, matching the goal
    tile's own distance, the same way route2_potential gives the
    forest's south gate map -ROUTE_2_GOAL_Y rather than the generic
    anchor -- continuity across the boundary, so stepping onto the goal
    is never scored as a large penalty relative to standing right next
    to it. Anywhere else off the forest (the 8 exits back to Route 2, or
    a blackout to a Pokemon Center) gets the worst-case anchor, the same
    idea as Route 1/Route 2's off-route case.
    """
    if position["map_id"] == CONNECTOR_MAP_ID:
        return 0
    if position["map_id"] != FOREST_MAP_ID:
        return -_MAX_DISTANCE

    tile = (position["x"], position["y"])
    return -_DISTANCES.get(tile, _MAX_DISTANCE)


def calculate_forest_reward(before, after):
    """
    Same shape as Route 1/Route 2's reward: a small step cost, a bigger
    cost for not moving at all, potential-based shaping toward the goal,
    a large terminal reward for reaching it, and a penalty for leaving
    the forest anywhere else -- which also covers losing a forced
    trainer battle, since a blackout lands on a Pokemon Center's map,
    neither the forest nor the connector room.
    """
    reward = -0.01

    if position_key(before) == position_key(after):
        reward -= 0.25

    reward += forest_potential(after) - forest_potential(before)

    if after["map_id"] == CONNECTOR_MAP_ID:
        reward += 100.0
    elif after["map_id"] not in (FOREST_MAP_ID, CONNECTOR_MAP_ID):
        reward -= 20.0

    return reward
