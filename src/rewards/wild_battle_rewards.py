# Reward shaping for wild Pokemon encounters -- extends battle_rewards.py's
# design (reward proportional to HP fraction dealt/lost each turn, plus a
# terminal bonus/penalty deliberately larger than anything the per-turn
# shaping could add up to) with the one thing a wild encounter has that the
# rival fight doesn't: the battle can end with *neither* side's HP at zero,
# because running away is actually legal here.

STEP_PENALTY = -0.01
INVALID_MOVE_PENALTY = -0.05

WIN_REWARD = 10.0
LOSS_REWARD = -10.0

# Deliberately smaller than WIN_REWARD: ending the battle safely (whether
# the player successfully ran, or the wild Pokemon fled on its own -- both
# produce the same "in_battle became False, no HP hit zero" signature, and
# there's no need to tell them apart) shouldn't be punished, but winning
# should still be what the agent prefers whenever it actually can.
FLED_REWARD = 1.0


def calculate_wild_battle_reward(before, after, invalid_action=False):
    """
    before/after are battle-state dicts from memory.get_wild_battle_state().

    invalid_action means the chosen move slot wasn't usable (unknown move
    or 0 PP) -- no button was pressed, so `before` and `after` are the
    same state. Choosing to run is never invalid regardless of PP/moves.
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
        else:
            reward += FLED_REWARD

    return reward
