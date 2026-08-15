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
# 12.0 makes one map boundary worth 200 tiles of wandering (200 x 0.06),
# and three story flags -- a real find. It stays well under a badge (60)
# and delivery (100) so it cannot outbid actual progress, and raising
# NEW_TILE_REWARD instead cannot do this job: explore is ~11% of episode
# reward, and any per-tile value high enough to matter would swamp
# everything else long before it pointed anywhere specific.
#
# Same non-farmable shape as every other term here: the env tracks the set
# of maps visited this episode (seeded with the starting map, exactly like
# visited_tiles), so a boundary pays once and re-entry pays nothing --
# door-hopping between two maps nets one payment per map per episode, ever.
NEW_MAP_REWARD = 12.0

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
# trip worth it -- see carry_home below for the other half of this fix.
MILESTONE_PICKUP_REWARD = 15.0
MILESTONE_DELIVERED_REWARD = 100.0

# Map-level "distance home" for the carry-home shaping term below. Hop counts
# to OAKS_LAB (map 40), by number of connections/warps crossed -- not
# guessed, but built from pret/pokered's own map headers and warp tables
# (data/maps/headers/*.asm's `connection` lines for outdoor maps,
# data/maps/objects/*.asm's `warp_event`/LAST_MAP pairs for building
# interiors), read directly rather than assumed from general game knowledge:
#
#   38 (RedsHouse2F, the bedroom) -- warps to --> 37 (RedsHouse1F)
#   37 (RedsHouse1F)              -- warps to --> 0  (PalletTown)
#   39 (BluesHouse)               -- warps to --> 0  (PalletTown)
#   40 (OaksLab)                  -- warps to --> 0  (PalletTown)
#   0  (PalletTown)               -- connects to --> 12 (Route1)
#   12 (Route1)                   -- connects to --> 1  (ViridianCity)
#   1  (ViridianCity)             -- warps to --> 41 (ViridianPokecenter),
#                                                 42 (ViridianMart),
#                                                 43 (ViridianSchoolHouse),
#                                                 44 (ViridianNicknameHouse)
#   1  (ViridianCity)             -- connects to --> 33 (Route22)
#   33 (Route22)                  -- warps to --> 193 (Route22Gate)
#
# Coarser than true tile distance -- it only changes when the agent crosses a
# map boundary, not while walking around inside one -- but it needs no new
# world-space survey data (only Route1/Viridian City have verified world
# offsets; see screenshots/world_atlas_meta.json) and it already separates
# exactly the ground this project's own eval data shows the agent
# over-exploring once it's holding the Parcel: Route 22 sits at hop 4, the
# same distance as every Viridian building, and further than anywhere on the
# direct Pallet-Route1-Viridian line home.
#
# Anything not listed (unexplored ground, or a map this table hasn't been
# extended to yet) falls back to UNKNOWN_HOP_DISTANCE, one hop past the
# farthest verified entry -- deliberately treated as "far", since the whole
# point of this term is to counterweight the pull of fresh exploration while
# the agent is supposed to be heading home instead.
OAKS_LAB_MAP_ID = 40
MAP_HOP_DISTANCE_TO_OAKS_LAB = {
    40: 0,    # OaksLab
    0: 1,     # PalletTown
    37: 2,    # RedsHouse1F
    39: 2,    # BluesHouse
    12: 2,    # Route1
    38: 3,    # RedsHouse2F (bedroom)
    1: 3,     # ViridianCity
    41: 4,    # ViridianPokecenter
    42: 4,    # ViridianMart
    43: 4,    # ViridianSchoolHouse
    44: 4,    # ViridianNicknameHouse
    33: 4,    # Route22
    193: 5,   # Route22Gate
}
UNKNOWN_HOP_DISTANCE = max(MAP_HOP_DISTANCE_TO_OAKS_LAB.values()) + 1

# Same weight class as EVENT_FLAG_REWARD -- closing one hop toward home while
# carrying the Parcel is treated as roughly as meaningful as flipping a story
# flag. Applied only on steps where the agent is holding a milestone item on
# *both* sides of the step (see calculate_whole_game_reward) -- the pickup
# and delivery steps themselves are scored solely by MILESTONE_PICKUP_REWARD
# and MILESTONE_DELIVERED_REWARD above, so this term never fires on the two
# steps that already have their own reward and can't be gamed by pickup/drop
# cycling. Within a single map it's silent (hop distance only changes on a
# map crossing), and any back-and-forth between two adjacent maps nets
# exactly zero over a round trip -- one hop of credit out, one hop of debit
# back -- so there's no loop to farm, the same non-exploitable shape every
# other delta-based term in this file already has.
CARRY_HOME_SHAPING_PER_HOP = 4.0

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

    # Dense signal for the leg the milestone reward alone couldn't teach:
    # carrying the Parcel home. Only on steps holding a milestone item both
    # before and after -- see CARRY_HOME_SHAPING_PER_HOP's comment for why
    # that boundary matters and why it can't be farmed.
    if before["milestone_items"] and after["milestone_items"]:
        components["carry_home"] = CARRY_HOME_SHAPING_PER_HOP * (
            MAP_HOP_DISTANCE_TO_OAKS_LAB.get(before["map_id"], UNKNOWN_HOP_DISTANCE)
            - MAP_HOP_DISTANCE_TO_OAKS_LAB.get(after["map_id"], UNKNOWN_HOP_DISTANCE)
        )
    else:
        components["carry_home"] = 0.0

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
