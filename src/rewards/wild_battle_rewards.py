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

# Deliberately the single largest terminal reward available: catching a
# wild Pokemon is a permanent addition to the party, worth strictly more
# than the XP from just fainting it -- teaching the agent that reaching
# for a Poke Ball is the *preferred* outcome when one is available, not
# merely tolerated. Set above WIN_REWARD rather than beside it so a
# state where both are reachable doesn't leave the agent indifferent
# between them.
CATCH_REWARD = 15.0


def calculate_wild_battle_reward(before, after, invalid_action=False, caught=False):
    """
    before/after are battle-state dicts from memory.get_wild_battle_state().

    invalid_action means the chosen move slot wasn't usable (unknown move
    or 0 PP) -- no button was pressed, so `before` and `after` are the
    same state. Choosing to run is never invalid regardless of PP/moves.

    caught means this exact step's action grew the party (checked by the
    caller via get_party_count before/after, the only signal that can't
    be confused with the battle ending some other way) -- a caught wild
    Pokemon leaves enemy_mon_hp > 0 and battle_mon_hp > 0 exactly like a
    successful flee does, so without this flag a catch would silently
    score as FLED_REWARD instead of the much larger, deliberately
    distinct bonus it's meant to teach.
    """

    # Added, not returned on its own. This originally returned the
    # penalty directly, from back when an invalid pick pressed no buttons
    # and left the state untouched -- but the environment now substitutes
    # the first valid move and really plays it (to avoid the deadlock the
    # rival battle env hit), so the turn has real consequences that still
    # need scoring. Returning early threw those away, meaning a battle
    # *won* on a turn where the agent named an unusable slot paid -0.05
    # instead of +10. Found by a smoke test of the trainer battle
    # environment, which shares this reward shape.
    reward = STEP_PENALTY
    if invalid_action:
        reward += INVALID_MOVE_PENALTY

    enemy_max_hp = max(before["enemy_mon_max_hp"], 1)
    your_max_hp = max(before["battle_mon_max_hp"], 1)

    damage_dealt_fraction = (before["enemy_mon_hp"] - after["enemy_mon_hp"]) / enemy_max_hp
    damage_taken_fraction = (before["battle_mon_hp"] - after["battle_mon_hp"]) / your_max_hp

    reward += damage_dealt_fraction
    reward -= damage_taken_fraction

    if not after["in_battle"]:
        if caught:
            reward += CATCH_REWARD
        elif after["enemy_mon_hp"] == 0:
            reward += WIN_REWARD
        elif after["battle_mon_hp"] == 0:
            reward += LOSS_REWARD
        else:
            reward += FLED_REWARD

    return reward
