# Reward shaping for trainer battles.
#
# Same family as battle_rewards.py and wild_battle_rewards.py -- reward
# proportional to the fraction of max HP dealt/lost each turn, a small
# per-turn penalty, and one terminal bonus deliberately larger than
# anything the per-turn shaping could accumulate, so prolonging a
# winnable fight never beats finishing it.
#
# Two things make trainers different from both earlier battle tasks:
#
#   - There is no running away. wild_battle_rewards.py's FLED_REWARD has
#     no counterpart here, and the environment offers no run action at
#     all, because Gen 1 simply refuses to let you flee a trainer.
#   - A trainer can have more than one Pokemon. When one faints, the next
#     is sent out and enemy HP jumps straight back to full. Read naively
#     as "damage dealt", that jump is a large negative reward for the one
#     turn the agent did best on -- it would actively teach the agent not
#     to knock anything out. So a change in the enemy's *species* is
#     treated as what it actually is: a knockout, worth a bonus, with the
#     HP delta for that turn ignored.

STEP_PENALTY = -0.01
INVALID_MOVE_PENALTY = -0.05

WIN_REWARD = 10.0
LOSS_REWARD = -10.0

# Paid per Pokemon knocked out, so progress through a multi-Pokemon
# trainer is rewarded as it happens rather than only at the very end.
# Deliberately well under WIN_REWARD: finishing the whole battle stays
# worth more than any single knockout.
KNOCKOUT_REWARD = 2.0


def calculate_trainer_battle_reward(before, after, invalid_action=False):
    """
    before/after are battle-state dicts from
    memory.get_detailed_battle_state() -- the detailed variant, because
    the enemy's species is needed to spot a replacement being sent out.
    """

    # The penalty is *added*, never returned on its own. An earlier
    # version of this family of reward functions returned it directly,
    # which quietly discarded everything else that happened on that
    # turn -- including winning. The environment substitutes the first
    # valid move and really plays it when an unusable slot is picked, so
    # the turn has genuine consequences that still have to be scored;
    # the penalty is for the bad *choice*, not a replacement for the
    # outcome. Caught by a smoke test showing won=True steps paying
    # -0.05 instead of +10.
    reward = STEP_PENALTY
    if invalid_action:
        reward += INVALID_MOVE_PENALTY

    enemy_max_hp = max(before["enemy_mon_max_hp"], 1)
    your_max_hp = max(before["battle_mon_max_hp"], 1)

    knocked_one_out = (
        after["in_battle"]
        and after["enemy_mon_species"] != before["enemy_mon_species"]
    )

    if knocked_one_out:
        reward += KNOCKOUT_REWARD
    else:
        reward += (before["enemy_mon_hp"] - after["enemy_mon_hp"]) / enemy_max_hp

    reward -= (before["battle_mon_hp"] - after["battle_mon_hp"]) / your_max_hp

    if not after["in_battle"]:
        # A trainer battle can only end one of two ways -- their whole
        # party is down, or ours is. Checking our own HP is the reliable
        # test: the enemy's HP field belongs to whichever Pokemon was
        # last on the field, which says nothing about the rest of a
        # multi-Pokemon party.
        if after["battle_mon_hp"] > 0:
            reward += WIN_REWARD
        else:
            reward += LOSS_REWARD

    return reward
