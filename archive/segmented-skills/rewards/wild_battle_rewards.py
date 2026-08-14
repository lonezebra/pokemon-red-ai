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

# A new species catching is deliberately the single largest terminal
# reward available -- a permanent addition to the collection, worth
# strictly more than the XP from just fainting it, so the agent learns
# reaching for a Poke Ball is *preferred* when it grows the collection.
#
# A *duplicate* species catching is a much smaller reward, deliberately
# below WIN_REWARD rather than anywhere near it. Gen 1 awards no XP for
# a capture (only a knockout does), so a duplicate catch is a real
# resource spent (one of a handful of Poke Balls this episode) for a
# Pokemon already represented, no XP, and no Pokedex progress -- the
# first version of this reward gave every catch the same large bonus
# regardless, which would have trained the agent to try catching every
# single encounter forever, full collection or not. Left positive
# rather than zero/negative: a duplicate isn't worthless (better IVs,
# evolution fodder, trade value all exist in the real game even though
# none of that is modeled here), just clearly the lesser choice next to
# either winning or catching something new.
CATCH_REWARD = 15.0
CATCH_DUPLICATE_REWARD = 2.0


def calculate_wild_battle_reward(
    before, after, invalid_action=False, caught=False, caught_new_species=False
):
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
    score as FLED_REWARD instead of one of the two bonuses below.

    caught_new_species further distinguishes which of those two bonuses
    applies -- see CATCH_DUPLICATE_REWARD for why a catch's value isn't
    flat. Meaningless when caught is False.
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
            reward += CATCH_REWARD if caught_new_species else CATCH_DUPLICATE_REWARD
        elif after["enemy_mon_hp"] == 0:
            reward += WIN_REWARD
        elif after["battle_mon_hp"] == 0:
            reward += LOSS_REWARD
        else:
            reward += FLED_REWARD

    return reward
