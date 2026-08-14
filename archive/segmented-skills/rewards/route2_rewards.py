ROUTE_2_MAP_ID = 13
VIRIDIAN_CITY_MAP_ID = 1

# The Viridian Forest south gate -- the goal. Established by an
# exhaustive survey of Route 2 (core.pathfind.survey_map, which
# flood-fills the whole map and finished with its frontier exhausted, so
# this really is the complete exit list rather than the best a random
# walk managed to find):
#
#   from (7,71) / (8,71) / (9,71) going down -> map 1  (back to Viridian)
#   from (3,44)          going up   -> map 50 (forward)
#
# Exactly one forward exit, so there is no ambiguity about what "done"
# means here. Contrast the previous attempt at a Route 2 task, whose
# success condition had to be the vague "any map that isn't this one or
# Viridian" because nobody had established where the route actually
# went -- and which then counted walking into a building as a win.
VIRIDIAN_FOREST_GATE_MAP_ID = 50

# Route 2 is entered at its southern end (y=71) and the forward exit is
# at y=44, so progress means *decreasing* y -- the same orientation as
# Route 1 (entry y=35, goal y=0), which is why the same -y potential
# works unchanged. Verified rather than assumed this time: the survey
# above is what fixes which direction is forward.
ROUTE_2_START_Y = 71
ROUTE_2_GOAL_Y = 44


def position_key(position):
    return (
        position["map_id"],
        position["x"],
        position["y"],
    )


def route2_potential(position):
    """
    Higher = closer to the goal. See rewards/route1_rewards.py for why
    this is potential-based shaping rather than a per-episode "new tile"
    bonus: that bonus depended on each episode's own visitation history,
    which made the same (state, action) pair worth different amounts in
    different episodes -- a noisy target for tabular Q-learning, and the
    direct cause of a policy that burned an entire 800-step budget
    revisiting 24 tiles.

    Anchored to the starting potential when off Route 2 entirely, for the
    same reason Route 1's is: Viridian City has its own unrelated local
    coordinates, so shaping on its y would be shaping on an arbitrary
    number. Leaving the route backward contributes no shaping either way
    -- the explicit penalty in calculate_route2_reward already makes that
    outcome clearly bad.
    """

    if position["map_id"] == VIRIDIAN_FOREST_GATE_MAP_ID:
        return -ROUTE_2_GOAL_Y
    if position["map_id"] == ROUTE_2_MAP_ID:
        return -position["y"]

    return -ROUTE_2_START_Y


def calculate_route2_reward(before, after):
    """
    Same shape as the Route 1 reward: a small step cost, a bigger cost
    for not moving at all, potential-based shaping toward the goal, a
    large terminal reward for reaching it, and a penalty for leaving the
    route backward.
    """

    reward = -0.01

    if position_key(before) == position_key(after):
        reward -= 0.25

    reward += route2_potential(after) - route2_potential(before)

    if after["map_id"] == VIRIDIAN_FOREST_GATE_MAP_ID:
        reward += 100.0
    elif after["map_id"] not in (ROUTE_2_MAP_ID, VIRIDIAN_FOREST_GATE_MAP_ID):
        reward -= 20.0

    return reward
