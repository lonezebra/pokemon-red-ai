BEDROOM_MAP_ID = 38
DOWNSTAIRS_MAP_ID = 37
PALLET_TOWN_MAP_ID = 0


def position_key(position):
    """
    Turn a position dictionary into a hashable key.

    Example:
        {"map_id": 38, "x": 3, "y": 6}
    becomes:
        (38, 3, 6)
    """

    return (
        position["map_id"],
        position["x"],
        position["y"],
    )


def calculate_leave_house_reward(before, after, visited_positions):
    """
    Reward function for the leave-house task.

    The goal is to reach Pallet Town, map 0.

    Reward design:
      - Small penalty every step so the agent prefers shorter paths.
      - Penalty if it bumps into something and does not move.
      - Small reward for visiting a new position.
      - Bigger reward for reaching downstairs.
      - Large reward for reaching Pallet Town.
    """

    reward = -0.01

    before_key = position_key(before)
    after_key = position_key(after)

    moved = before_key != after_key

    if not moved:
        reward -= 0.25

    if after_key not in visited_positions:
        reward += 1.0

    # Checking before != after here matters: reaching downstairs doesn't end
    # the episode (unlike Pallet Town below), so checking current map alone
    # would pay out +5 on every single step the agent lingers on that map --
    # discovered by noticing training rewards were far higher than the
    # reward design should ever allow in a 200-step episode.
    if after["map_id"] == DOWNSTAIRS_MAP_ID and before["map_id"] != DOWNSTAIRS_MAP_ID:
        reward += 5.0

    if after["map_id"] == PALLET_TOWN_MAP_ID:
        reward += 100.0

    return reward