ROUTE_1_MAP_ID = 12
VIRIDIAN_CITY_MAP_ID = 1

# The Route 1 entry state (saves/route_1_entry.state) starts at y=35;
# Viridian City's entrance is reached around y=0. Used as the "worst
# case" anchor for route1_potential() below when the player has left
# Route 1 entirely (see there for why).
ROUTE_1_START_Y = 35


def position_key(position):
    return (
        position["map_id"],
        position["x"],
        position["y"],
    )


def route1_potential(position):
    """
    A pure function of position, higher = closer to the goal, used for
    potential-based reward shaping below. Route 1 is a mostly-vertical
    corridor (confirmed by the mapping scout in build_route1_map.py:
    entry at y=35, Viridian City's end at y=0), so y-coordinate alone is
    a reasonable stand-in for "distance to the goal," without needing
    real pathfinding.

    Deliberately NOT defined in terms of Pallet Town's own y when the
    player has left Route 1 backward -- Pallet Town (map_id 0) has its
    own unrelated local (x, y) grid (a hard-learned lesson from the
    Pallet Town reward-exploit bug), so using its y here would add a
    shaping term based on essentially arbitrary numbers. Anchoring it to
    the Route 1 starting potential instead means leaving the route
    contributes no shaping bonus or penalty of its own -- the explicit
    -20 in calculate_route1_reward already makes that outcome clearly
    bad on its own.
    """

    if position["map_id"] == VIRIDIAN_CITY_MAP_ID:
        return 0.0
    if position["map_id"] == ROUTE_1_MAP_ID:
        return -position["y"]

    return -ROUTE_1_START_Y


def calculate_route1_reward(before, after):
    """
    Reward function for the Route 1 navigation task.

    The goal is to reach Viridian City, map 1, starting from
    saves/route_1_entry.state (Route 1, map 12).

    Wild Pokemon encounters are handled by the environment itself before
    this function ever runs (see envs/route1_env.py), by automatically
    attempting to run away -- so from this reward function's point of
    view, `before` and `after` are always overworld positions, never a
    battle state.

    This used to give +1.0 for visiting any tile not already in
    `visited_positions` this episode. That turned out to have a subtle
    problem going deeper than the earlier Pallet-Town exploit (see
    ROUTE_1_START_Y's docstring and the git history for that one): since
    `visited_positions` resets every episode, whether a given (state,
    action) pair earned that bonus depended on each *episode's own*
    visitation history, not on the game state alone -- the same
    transition could be rewarded in one training episode and not in
    another. Averaged tabularly over thousands of episodes, that's a
    noisy, inconsistent target for Q-learning to fit, and it showed:
    diagnosed directly by recording a full greedy playthrough of the
    trained policy, which ran the entire 800-step budget but visited
    only 24 distinct tiles, spending 777 of 801 steps revisiting
    ground it had already covered -- a directionless revisit loop, not
    genuine progress cut short by running out of steps.

    Potential-based shaping (Ng, Harada & Russell 1999) fixes this: the
    bonus below is `route1_potential(after) - route1_potential(before)`,
    a plain function of position with no episode-local history involved,
    so the same transition always earns the same shaping reward. It's
    also a strictly stronger signal than the old bonus -- moving away
    from the goal is now an explicit penalty (symmetric with the reward
    for moving toward it), rather than merely forfeiting a bonus.
    """

    reward = -0.01

    before_key = position_key(before)
    after_key = position_key(after)

    moved = before_key != after_key

    if not moved:
        reward -= 0.25

    reward += route1_potential(after) - route1_potential(before)

    if after["map_id"] == VIRIDIAN_CITY_MAP_ID:
        reward += 100.0
    elif after["map_id"] not in (ROUTE_1_MAP_ID, VIRIDIAN_CITY_MAP_ID):
        reward -= 20.0

    return reward
