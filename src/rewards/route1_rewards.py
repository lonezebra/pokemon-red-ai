ROUTE_1_MAP_ID = 12
VIRIDIAN_CITY_MAP_ID = 1


def position_key(position):
    return (
        position["map_id"],
        position["x"],
        position["y"],
    )


def calculate_route1_reward(before, after, visited_positions):
    """
    Reward function for the Route 1 navigation task.

    The goal is to reach Viridian City, map 1, starting from
    saves/route_1_entry.state (Route 1, map 12). Same shape as the
    leave-house reward (small step penalty, penalty for not moving,
    reward for visiting a new tile, large reward for the goal) --
    Route 1 is a much longer corridor than the bedroom-to-Pallet-Town
    walk, but it's still fundamentally the same kind of task: get from
    a known start to a known goal map, so the same design carries over
    directly.

    Wild Pokemon encounters are handled by the environment itself before
    this function ever runs (see envs/route1_env.py), by automatically
    attempting to run away -- so from this reward function's point of
    view, `before` and `after` are always overworld positions, never a
    battle state.

    Discovered the hard way (a trained agent's greedy policy turned out
    to walk straight back into Pallet Town on its very first move, then
    stay there): the "new tile" bonus below didn't check *which* map the
    new tile was on, so wandering backward into Pallet Town earned the
    same +1.0 novelty reward as real forward progress on Route 1 --
    an easier way to farm reward than pushing through Route 1's grass
    toward a goal never yet reached. Route1Env now ends the episode the
    moment the player leaves Route 1 backward (see reached_goal handling
    there), and the big penalty below makes sure that ending reads as
    unambiguously bad rather than just "no bonus this step."
    """

    reward = -0.01

    before_key = position_key(before)
    after_key = position_key(after)

    moved = before_key != after_key

    if not moved:
        reward -= 0.25

    if after["map_id"] == ROUTE_1_MAP_ID and after_key not in visited_positions:
        reward += 1.0

    if after["map_id"] == VIRIDIAN_CITY_MAP_ID:
        reward += 100.0
    elif after["map_id"] not in (ROUTE_1_MAP_ID, VIRIDIAN_CITY_MAP_ID):
        reward -= 20.0

    return reward
