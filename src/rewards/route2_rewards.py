VIRIDIAN_CITY_MAP_ID = 1
ROUTE_2_MAP_ID = 33


def position_key(position):
    return (
        position["map_id"],
        position["x"],
        position["y"],
    )


def calculate_route2_reward(before, after, visited_positions):
    """
    Reward function for the Route 2 navigation task.

    Unlike Route 1 (a clean, mostly-vertical corridor where progress
    toward the goal could be shaped directly from the y-coordinate),
    Route 2's actual layout and exact exit toward Viridian Forest wasn't
    pinned down by scripted scouting -- several thousand steps of biased
    random walking from saves/route2_entry.state explored a wide area
    (x roughly 2-39, y roughly 6-15) without finding it. Rather than
    keep scouting by hand indefinitely, this reward function goes back
    to the simpler per-episode novelty bonus Route 1 originally used
    (before potential-based shaping replaced it -- see route1_rewards.py
    for why that turned out to be a noisy Q-learning target) and leans
    on the agent's own much more extensive training-time exploration to
    actually find the way through, the same way leave-house and Route
    1's own first successes did before any shaping existed.

    This is a real, known risk carried over from Route 1's history, not
    an oversight: if training shows the same directionless-revisit-loop
    instability Route 1 hit, the fix is the same one that worked there
    (potential-based shaping) -- it just needs a known distance metric
    first, which requires knowing where the actual goal is. Once
    training finds it even once, that trajectory is exactly the
    information needed to build one.
    """

    reward = -0.01

    before_key = position_key(before)
    after_key = position_key(after)

    moved = before_key != after_key

    if not moved:
        reward -= 0.25

    if after["map_id"] == ROUTE_2_MAP_ID and after_key not in visited_positions:
        reward += 1.0

    if after["map_id"] not in (ROUTE_2_MAP_ID, VIRIDIAN_CITY_MAP_ID):
        # Any map beyond Route 2 itself counts as forward progress --
        # deliberately not naming Viridian Forest's map ID specifically,
        # since scouting never actually confirmed it.
        reward += 100.0
    elif after["map_id"] == VIRIDIAN_CITY_MAP_ID:
        reward -= 20.0

    return reward
