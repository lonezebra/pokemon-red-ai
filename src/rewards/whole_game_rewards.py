"""
Scoring for the whole-game agent.

Every other reward module in this project scores one task against one known
goal tile or one battle outcome, because every other task in this project was
scoped small enough to *have* one. This one can't: the agent starts in the
bedroom with no goal coordinate, and "the goal" changes continuously as the
game opens up. So the shape is different on purpose.

The rule here is that reward comes from *things the game itself recorded as
progress* -- badges earned, story flags set, levels gained, new ground
covered -- read straight out of memory, rather than from anything this
project hand-picked as the next milestone. That is what makes it a
whole-game reward instead of a very long list of per-route rewards.

All terms are deltas between consecutive steps. Absolute values are never
scored, which matters more than it sounds: rewarding an absolute (say, "you
have 2 badges") pays the agent every single step for something it did once,
which is precisely the loitering bug leave_house_rewards.py had to fix (see
README.md) -- there the +5 for being downstairs was paid per step instead of
on arrival, and the agent learned to stand there collecting it.
"""

from core.memory import OAKS_PARCEL_ITEM_ID

# A badge is the single least ambiguous statement the game makes about real
# progress, and there are only eight in the entire game, so this is scored
# far above everything else. It cannot be farmed -- the delta is nonzero
# exactly once per badge, ever.
BADGE_REWARD = 60.0

# Story flags advance for beating a rival, clearing a blocked door, being
# handed a key item. Dense enough to guide, and like badges they only ever
# flip once each.
EVENT_FLAG_REWARD = 4.0

# Levels are the one progress signal an agent can genuinely grind, so they
# are worth real reward (a levelled party is what unblocked the trainer
# battles in this project) but not enough to make grinding beat exploring.
LEVEL_REWARD = 1.0

# New ground, keyed by (map_id, x, y). The main driver early on, when the
# agent has no idea badges or events exist. Small per tile because there are
# thousands of them.
#
# PWhiddy's v1 used a KNN over recent *frames* to decide novelty and v2
# replaced it with exactly this coordinate-based version; coordinates are
# cheaper, need no frame buffer, and here they come from memory addresses
# this project verified long ago.
NEW_TILE_REWARD = 0.06

# Entering a map not yet visited this episode. Added at 330M steps, when the
# per-tile term above was measured to be the binding constraint on progress:
# with delivery solved (36/36), the policy still spent recovered episode
# time re-walking known ground, because episodic tile novelty pays the same
# +0.06 for the thousandth lap of Viridian as for the first step into Route
# 2 -- there is no gradient pointing at the frontier at all. Forcibly
# unsticking the agent proved the point without any retraining
# (tools/measure_stall_breaker.py): a fifth of the episode handed back
# yielded +15% tiles, zero new events, and no new maps.
#
# 5.0, cut from the original 12.0 by the first live test of this term.
# At 12.0, eight reachable map boundaries added up to ~96 of dense, early,
# zero-risk income per episode, and 20M steps of retraining from the
# delivery-solved 330M checkpoint converged on collecting exactly that: the
# policy toured 7-8 maps per episode (Route 22 went from never-visited to
# 9.4% of all steps) and abandoned the Parcel errand completely -- 0/28
# deliveries, the Mart never entered, from a checkpoint that delivered
# 36/36. The gated maps the bonus was aimed at stayed locked, because
# unlocking them requires the delivery it out-competed.
#
# 5.0 keeps a real frontier pull -- one boundary still buys ~83 tiles of
# wandering, and slightly over one story flag -- while the full tour now
# totals ~40, well under the 115 the errand pays. Still far below a badge
# (60 for ONE find) and delivery (100), and raising NEW_TILE_REWARD instead
# still cannot do this job: any per-tile value loud enough to matter swamps
# everything else long before it points anywhere specific.
#
# Same non-farmable shape as every other term here: the env tracks the set
# of maps visited this episode (seeded with the starting map, exactly like
# visited_tiles), so a boundary pays once and re-entry pays nothing --
# door-hopping between two maps nets one payment per map per episode, ever.
NEW_MAP_REWARD = 5.0

# Healing is rewarded so that using a Pokemon Center is learnable rather than
# something the agent has to stumble into, but capped per step so it cannot
# be turned into an income stream by repeatedly taking and healing damage.
HEAL_REWARD = 3.0
MAX_HEAL_FRACTION_PER_STEP = 0.5

# Losing the whole party sends you back to a Pokemon Center having achieved
# nothing. Penalised, but mildly: this is a normal part of learning the game
# and an agent too afraid of it will never explore anywhere dangerous.
#
# Charged on the *transition* into a blackout, never for being in one. The
# first version of this scored the state instead, and the first real rollouts
# showed what that costs: a fainted party stays at zero HP for the hundreds
# of steps the game spends walking you to a Pokemon Center, so one blackout
# billed -5168 instead of -8 and drowned out every other term in the episode.
#
# This is the same mistake leave_house_rewards.py made with its downstairs
# bonus, in the same direction, and it is worth stating twice: score the
# event, not the condition.
BLACKOUT_PENALTY = -8.0

# Key items gated behind a specific NPC hand-off rather than anything
# explorable by walking -- across 160M+ steps and dozens of eval batches,
# this project's whole-game agent has never once shown Route 2, Viridian
# Forest, or Pewter City in a per-map breakdown, because Route 2's gate
# stays shut until Oak's Parcel is delivered (Viridian Mart -> walk back to
# Pallet Town -> give it to Oak), and nothing about that errand is
# reachable by the exploration reward alone: it means walking *away* from
# the frontier into ground that's already been mapped. This table exists so
# future milestones shaped like this one (an NPC favor that unlocks real
# progress, not just a new tile) can be added the same way -- verify the
# item id, add one line, done.
#
# OAKS_PARCEL_ITEM_ID's value was verified two ways before being trusted
# here, not assumed: it was already empirically checked against the real
# game once, by the old segmented-skills approach (see its comment in
# core/memory.py), and independently re-confirmed against pret/pokered's
# own constants/item_constants.asm (OAKS_PARCEL = $46 = 70 decimal) before
# this reward was written.
MILESTONE_ITEMS = {OAKS_PARCEL_ITEM_ID: "oaks_parcel"}
MILESTONE_ITEM_IDS = frozenset(MILESTONE_ITEMS)

# Pickup is worth roughly 4 event flags -- a real find, not a tile. Delivery
# is worth far more than that: it's what actually opens Route 2, Viridian
# Forest and Pewter City, so it's scored above even a badge (BADGE_REWARD),
# not just above the pickup that precedes it. This was raised from 25.0 after
# 150M+ steps of live data showed *why* it needed to be: pickup happens in
# almost every episode (the Mart is reachable by ordinary exploration), but
# delivery had happened exactly once across ten checkpoints' worth of eval
# batches. A 25.0 payoff, thousands of steps after the last dense reward and
# competing against a whole episode's worth of exploration reward for territory
# in the other direction, was not a strong enough signal to make the return
# trip worth it. (The other half of that fix was the carry_home shaping,
# since retired -- see the note directly below.)
MILESTONE_PICKUP_REWARD = 15.0
MILESTONE_DELIVERED_REWARD = 100.0

# carry_home -- the +/-4-per-hop shaping that guided the Parcel back to Oak
# while it was held -- was RETIRED here, deliberately, after doing its job
# and then turning hostile. It was added when delivery had happened once in
# ten checkpoints and the return trip needed a dense signal; by 330M steps
# the errand ran at 36/36 with the term essentially inert (a beeline home
# nets the same total as no term at all). Then NEW_MAP_REWARD arrived and
# the interaction bit: for a policy that also tours map boundaries, holding
# the Parcel turned every wander away from Oak into a -4-per-hop tax, and
# 20M steps of retraining found the obvious dodge -- never enter the Mart,
# never pick the Parcel up (0/28 deliveries from a 36/36 checkpoint, Mart
# untouched across every eval episode). A shaping term that punishes
# *starting* the errand it was built to finish has no remaining upside once
# the finished behavior exists in the resumed checkpoint; scaffolding comes
# down when the wall stands. The hop table it used (built from pret/pokered
# map headers, not guessed) lives on in git history should some future
# milestone need distance-to-target shaping again.

# Deliberately almost nothing, and worth explaining rather than tuning by
# feel. Episodes here always run the full max_steps -- Pokemon Red has no
# losing end state, so nothing ever terminates early -- which means the total
# step cost is identical for every possible policy and cannot change which
# one is best. It is a constant offset, not an incentive.
#
# The first value (-0.008) made that harmless constant the loudest number in
# the episode: -65.5 against +15.4 for exploring 257 new tiles, which made
# the logs read as though exploring were a losing move. Kept at a token
# value only so the term still exists if episodes ever gain a real
# termination condition, where it would start to mean something.
STEP_COST = -0.001


def calculate_whole_game_reward(before, after, tile_is_new, map_is_new=False):
    # map_is_new defaults to False -- the neutral "term doesn't fire" value
    # -- so the twenty-odd single-term checks in
    # tools/verify_whole_game_rewards.py don't each carry a fourth argument
    # that is noise for what they test. The env always passes it explicitly.
    """
    Score one step from two memory snapshots (see WholeGameEnv._read_state)
    plus whether the tile just entered had never been visited this episode.

    Returns (total_reward, components) -- the per-term breakdown comes back
    alongside the total because the failure mode this project keeps hitting
    is a reward that trains something other than what was intended, and an
    aggregate number alone never shows that. The env forwards it into `info`
    so it lands in the training logs.
    """
    components = {}

    components["badge"] = BADGE_REWARD * max(
        0, after["badges"] - before["badges"]
    )

    components["event"] = EVENT_FLAG_REWARD * max(
        0, after["events"] - before["events"]
    )

    # Summed across the party, so evolving or catching something new also
    # registers rather than only the lead Pokemon's growth.
    components["level"] = LEVEL_REWARD * max(
        0, sum(after["levels"]) - sum(before["levels"])
    )

    components["explore"] = NEW_TILE_REWARD if tile_is_new else 0.0

    # The frontier term: crossing into a map this episode hasn't seen. The
    # env decides map_is_new the same way it decides tile_is_new (a
    # per-episode visited set, seeded at reset), so this fires exactly once
    # per map per episode and door-hopping between two maps farms nothing.
    components["new_map"] = NEW_MAP_REWARD if map_is_new else 0.0

    # Fires exactly once per pickup and once per delivery -- set difference
    # against the previous step's held items, same non-farmable shape as
    # the badge/event terms above (an item sitting in the bag for the next
    # several thousand steps pays nothing further).
    picked_up = after["milestone_items"] - before["milestone_items"]
    delivered = before["milestone_items"] - after["milestone_items"]
    components["milestone"] = (
        MILESTONE_PICKUP_REWARD * len(picked_up)
        + MILESTONE_DELIVERED_REWARD * len(delivered)
    )

    # No carry_home term here anymore -- holding the Parcel is free, in
    # every direction. See the retirement note below
    # MILESTONE_DELIVERED_REWARD for what it did and why it had to go.

    healed = after["hp_fraction"] - before["hp_fraction"]
    if healed > 0 and not after["blacked_out"]:
        components["heal"] = HEAL_REWARD * min(healed, MAX_HEAL_FRACTION_PER_STEP)
    else:
        components["heal"] = 0.0

    # Edge, not level -- see BLACKOUT_PENALTY. Nonzero only on the single
    # step the party actually goes down.
    just_blacked_out = after["blacked_out"] and not before["blacked_out"]
    components["blackout"] = BLACKOUT_PENALTY if just_blacked_out else 0.0

    components["step"] = STEP_COST

    return sum(components.values()), components
