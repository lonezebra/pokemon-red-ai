# Reward shaping for the rival-battle task.
#
# Design (see project README roadmap): reward is proportional to the
# fraction of max HP dealt/lost each turn, plus a small per-turn penalty,
# plus one terminal reward that is deliberately much larger than anything
# the per-turn shaping could add up to -- this keeps the agent from ever
# preferring to prolong a winnable battle over just finishing it.

STEP_PENALTY = -0.01
INVALID_MOVE_PENALTY = -0.05

WIN_REWARD = 10.0
LOSS_REWARD = -10.0


def calculate_battle_reward(before, after, invalid_action=False):
    """
    before/after are battle-state dicts from memory.get_battle_state().

    invalid_action means the chosen move slot wasn't usable (unknown move
    or 0 PP) -- no button was pressed, so `before` and `after` are the
    same state.
    """

    if invalid_action:
        return INVALID_MOVE_PENALTY

    reward = STEP_PENALTY

    enemy_max_hp = max(before["enemy_mon_max_hp"], 1)
    your_max_hp = max(before["battle_mon_max_hp"], 1)

    damage_dealt_fraction = (before["enemy_mon_hp"] - after["enemy_mon_hp"]) / enemy_max_hp
    damage_taken_fraction = (before["battle_mon_hp"] - after["battle_mon_hp"]) / your_max_hp

    reward += damage_dealt_fraction
    reward -= damage_taken_fraction

    if not after["in_battle"]:
        if after["enemy_mon_hp"] == 0:
            reward += WIN_REWARD
        elif after["battle_mon_hp"] == 0:
            reward += LOSS_REWARD

    return reward
