# Memory addresses for Pokemon Red / Blue.
# These are in Game Boy WRAM.

ADDR_MAP_ID = 0xD35E
ADDR_PLAYER_Y = 0xD361
ADDR_PLAYER_X = 0xD362


def get_map_id(pyboy):
    return pyboy.memory[ADDR_MAP_ID]


def get_player_x(pyboy):
    return pyboy.memory[ADDR_PLAYER_X]


def get_player_y(pyboy):
    return pyboy.memory[ADDR_PLAYER_Y]


def get_player_position(pyboy):
    return {
        "map_id": get_map_id(pyboy),
        "x": get_player_x(pyboy),
        "y": get_player_y(pyboy),
    }


def print_player_position(pyboy, label="Position"):
    pos = get_player_position(pyboy)

    print()
    print(label)
    print("-" * len(label))
    print(f"Map ID: {pos['map_id']}")
    print(f"X:      {pos['x']}")
    print(f"Y:      {pos['y']}")


# Battle-related addresses.
#
# Sourced from the public pret/pokered disassembly (macros/ram.asm's
# battle_struct layout, anchored on the documented wBattleMonHP/wEnemyMonHP
# addresses), then empirically verified against saves/rival_battle.state:
# every value read from these addresses was cross-checked against the
# actual on-screen HP/PP numbers during the first rival battle before
# being trusted (see src/create_rival_battle_state.py).

ADDR_IS_IN_BATTLE = 0xD057  # 0 = not in battle, 1 = wild battle, 2 = trainer battle

ADDR_BATTLE_MON_HP = 0xD015      # 2 bytes, big-endian
ADDR_BATTLE_MON_MAX_HP = 0xD023  # 2 bytes, big-endian
ADDR_BATTLE_MON_MOVES = 0xD01C   # 4 bytes, one move ID per slot (0 = empty slot)
ADDR_BATTLE_MON_PP = 0xD02D      # 4 bytes, one PP value per move slot

ADDR_ENEMY_MON_HP = 0xCFE6
ADDR_ENEMY_MON_MAX_HP = 0xCFF4
ADDR_ENEMY_MON_MOVES = 0xCFED
ADDR_ENEMY_MON_PP = 0xCFFE

# Added for the wild-encounter milestone, needed because (unlike the
# rival's fixed Squirtle-vs-Bulbasaur matchup) the opponent's species
# actually varies. Species and level sit right before HP/MaxHP in the
# same battle_struct layout as everything above -- verified directly
# against a real Route 1 encounter, not just derived from the pattern:
# walked into the grass until a battle triggered, advanced to the
# FIGHT/PKMN/ITEM/RUN menu, and cross-checked these values against the
# actual on-screen "PIDGEY :L3" / player ":L6" text and Squirtle sprite
# (see src/create_wild_encounter_state.py). Species values are Gen 1's
# internal index order, not the National Pokedex number -- confirmed:
# 36 read here matched the on-screen PIDGEY, and Gen 1's internal index
# table independently lists Pidgey at 36.
ADDR_BATTLE_MON_SPECIES = 0xD014
ADDR_BATTLE_MON_LEVEL = 0xD022
ADDR_ENEMY_MON_SPECIES = 0xCFE5
ADDR_ENEMY_MON_LEVEL = 0xCFF3


# Bag contents. Stored as a count byte followed by (item id, quantity)
# pairs. Verified empirically before being relied on: read as empty
# standing outside the Viridian Mart, then read as exactly one entry --
# item 70, quantity 1 -- the instant the Mart clerk's script handed over
# Oak's Parcel, and empty again the instant Oak accepted it.
ADDR_NUM_BAG_ITEMS = 0xD31D
ADDR_BAG_ITEMS = 0xD31E
MAX_BAG_ITEMS = 20

OAKS_PARCEL_ITEM_ID = 70
POKE_BALL_ITEM_ID = 4

# Player's money, BCD-encoded across 3 bytes (each byte holds two decimal
# digits, e.g. 0x23 means "23" not 35). Verified against the actual Mart
# screen, not just the well-known disassembly address: 0xD347 read as
# [0x00, 0x23, 0x19] and the buy-menu's own on-screen total read ¥2319 --
# an exact match once decoded as BCD rather than raw binary.
ADDR_MONEY = 0xD347


def _bcd_to_int(byte):
    return (byte >> 4) * 10 + (byte & 0x0F)


def get_money(pyboy):
    return sum(
        _bcd_to_int(pyboy.memory[ADDR_MONEY + i]) * (100 ** (2 - i))
        for i in range(3)
    )


# The first party Pokemon, readable outside battle (the battle_struct
# addresses above only hold meaningful values mid-fight). Verified
# against known ground truth before use: standing in Viridian Forest
# with one Lv6 Squirtle on 23/23 HP, these read species 177, level 6,
# and 23/23 -- matching both the party count and the battle-struct
# values seen during the trainer captures.
ADDR_PARTY_COUNT = 0xD163
ADDR_PARTY_SPECIES = 0xD16B
ADDR_PARTY_HP = 0xD16C       # 2 bytes, big-endian
ADDR_PARTY_LEVEL = 0xD18C
ADDR_PARTY_MAX_HP = 0xD18D   # 2 bytes, big-endian


def read_u16(pyboy, addr):
    """Read a big-endian 2-byte value."""
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


def get_party_count(pyboy):
    return pyboy.memory[ADDR_PARTY_COUNT]


def get_party_level(pyboy):
    return pyboy.memory[ADDR_PARTY_LEVEL]


def get_party_hp(pyboy):
    return read_u16(pyboy, ADDR_PARTY_HP)


def get_party_max_hp(pyboy):
    return read_u16(pyboy, ADDR_PARTY_MAX_HP)


def get_party_hp_fraction(pyboy):
    return get_party_hp(pyboy) / max(get_party_max_hp(pyboy), 1)


def get_bag_item_ids(pyboy):
    count = min(pyboy.memory[ADDR_NUM_BAG_ITEMS], MAX_BAG_ITEMS)
    return [pyboy.memory[ADDR_BAG_ITEMS + i * 2] for i in range(count)]


def get_bag_item_quantity(pyboy, item_id):
    """0 if the item isn't held at all, matching has_item's convention
    of a plain boolean-ish read rather than raising on absence."""
    count = min(pyboy.memory[ADDR_NUM_BAG_ITEMS], MAX_BAG_ITEMS)
    for i in range(count):
        if pyboy.memory[ADDR_BAG_ITEMS + i * 2] == item_id:
            return pyboy.memory[ADDR_BAG_ITEMS + i * 2 + 1]
    return 0


def has_item(pyboy, item_id):
    return item_id in get_bag_item_ids(pyboy)


def is_in_battle(pyboy):
    return pyboy.memory[ADDR_IS_IN_BATTLE] != 0


def get_battle_mon_hp(pyboy):
    return read_u16(pyboy, ADDR_BATTLE_MON_HP)


def get_battle_mon_max_hp(pyboy):
    return read_u16(pyboy, ADDR_BATTLE_MON_MAX_HP)


def get_battle_mon_moves(pyboy):
    return [pyboy.memory[ADDR_BATTLE_MON_MOVES + i] for i in range(4)]


def get_battle_mon_pp(pyboy):
    return [pyboy.memory[ADDR_BATTLE_MON_PP + i] for i in range(4)]


def get_enemy_mon_hp(pyboy):
    return read_u16(pyboy, ADDR_ENEMY_MON_HP)


def get_enemy_mon_max_hp(pyboy):
    return read_u16(pyboy, ADDR_ENEMY_MON_MAX_HP)


def get_enemy_mon_moves(pyboy):
    return [pyboy.memory[ADDR_ENEMY_MON_MOVES + i] for i in range(4)]


def get_enemy_mon_pp(pyboy):
    return [pyboy.memory[ADDR_ENEMY_MON_PP + i] for i in range(4)]


def get_battle_mon_species(pyboy):
    return pyboy.memory[ADDR_BATTLE_MON_SPECIES]


def get_battle_mon_level(pyboy):
    return pyboy.memory[ADDR_BATTLE_MON_LEVEL]


def get_enemy_mon_species(pyboy):
    return pyboy.memory[ADDR_ENEMY_MON_SPECIES]


def get_enemy_mon_level(pyboy):
    return pyboy.memory[ADDR_ENEMY_MON_LEVEL]


def get_battle_type(pyboy):
    """0 = not in battle, 1 = wild battle, 2 = trainer battle."""
    return pyboy.memory[ADDR_IS_IN_BATTLE]


def get_battle_state(pyboy):
    """
    Minimal battle observation: own/enemy HP, and enough move/PP info to
    tell which moves are currently selectable (known move with PP > 0).
    """

    return {
        "in_battle": is_in_battle(pyboy),
        "battle_mon_hp": get_battle_mon_hp(pyboy),
        "battle_mon_max_hp": get_battle_mon_max_hp(pyboy),
        "battle_mon_moves": get_battle_mon_moves(pyboy),
        "battle_mon_pp": get_battle_mon_pp(pyboy),
        "enemy_mon_hp": get_enemy_mon_hp(pyboy),
        "enemy_mon_max_hp": get_enemy_mon_max_hp(pyboy),
    }


def get_detailed_battle_state(pyboy):
    """
    Same as get_battle_state(), plus the fields that only matter once
    the opponent is not always the same fixed matchup: battle_type
    (wild vs trainer) and both sides' species/level. Species in
    particular is what tells a trainer sending out a replacement --
    enemy HP jumping back to full -- apart from healing.
    """

    state = get_battle_state(pyboy)
    state["battle_type"] = get_battle_type(pyboy)
    state["battle_mon_species"] = get_battle_mon_species(pyboy)
    state["battle_mon_level"] = get_battle_mon_level(pyboy)
    state["enemy_mon_species"] = get_enemy_mon_species(pyboy)
    state["enemy_mon_level"] = get_enemy_mon_level(pyboy)
    return state


def print_battle_state(pyboy, label="Battle state"):
    state = get_battle_state(pyboy)

    print()
    print(label)
    print("-" * len(label))
    print(f"In battle:     {state['in_battle']}")
    print(f"Your HP:       {state['battle_mon_hp']}/{state['battle_mon_max_hp']}")
    print(f"Your moves/PP: {list(zip(state['battle_mon_moves'], state['battle_mon_pp']))}")
    print(f"Enemy HP:      {state['enemy_mon_hp']}/{state['enemy_mon_max_hp']}")


# Detecting "the battle is ready for the next player decision" (the
# FIGHT/ITEM/RUN menu is on screen) without guessing a fixed number of
# dialogue-advancing button presses.
#
# Different moves produce different amounts of text (e.g. Tail Whip adds
# a "The enemy's DEFENSE fell!" message that Tackle doesn't), so a fixed
# press count is unreliable -- this was discovered empirically while
# building the battle environment. Instead, we compare the window
# tilemap (where PyBoy renders text boxes/menus) against a known-good
# snapshot captured from a verified rival_battle.state, the same instant
# the FIGHT/ITEM/RUN menu is shown for the very first turn.

BATTLE_MENU_TILEMAP_ROWS = range(12, 18)

BATTLE_MENU_REFERENCE_TILEMAP = (
    (377, 378, 378, 378, 378, 378, 378, 378, 377, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 379),
    (380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380),
    (380, 383, 383, 383, 383, 383, 383, 383, 380, 237, 133, 136, 134, 135, 147, 383, 225, 226, 383, 380),
    (380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 383, 383, 383, 383, 383, 383, 383, 383, 383, 380),
    (380, 383, 383, 383, 383, 383, 383, 383, 380, 383, 136, 147, 132, 140, 383, 383, 145, 148, 141, 380),
    (381, 378, 378, 378, 378, 378, 378, 378, 381, 378, 378, 378, 378, 378, 378, 378, 378, 378, 378, 382),
)


def is_battle_menu_open(pyboy):
    """
    True exactly when the FIGHT/ITEM/RUN menu is on screen and the battle
    is waiting for the next player decision.
    """

    current = tuple(
        tuple(pyboy.tilemap_window[col, row] for col in range(20))
        for row in BATTLE_MENU_TILEMAP_ROWS
    )
    return current == BATTLE_MENU_REFERENCE_TILEMAP


# Reading which move is currently highlighted in the FIGHT move-select
# list. This turned out to matter a lot: the cursor is "sticky" (it
# remembers the last move used rather than resetting each turn), and the
# move list wraps around instead of clamping at the top/bottom -- both
# discovered empirically while building the battle environment. That
# combination makes a fixed sequence of up/down presses unreliable, so we
# read the cursor's actual row (marked by the "> " arrow tile, ID 237) in
# the window tilemap instead of assuming a starting position.

MOVE_CURSOR_ARROW_TILE_ID = 237
MOVE_CURSOR_COLUMN = 5
MOVE_CURSOR_ROWS = (13, 14, 15, 16)  # move slots 0-3


def get_move_cursor_slot(pyboy):
    """
    Return which move slot (0-3) the cursor is currently on, or None if
    the move list isn't open / the arrow can't be found.
    """

    for slot, row in enumerate(MOVE_CURSOR_ROWS):
        if pyboy.tilemap_window[MOVE_CURSOR_COLUMN, row] == MOVE_CURSOR_ARROW_TILE_ID:
            return slot

    return None


# Randomizing the player's starting battle stats, so the battle DQN has
# to generalize across different starter IV rolls instead of memorizing
# one exact matchup.
#
# create_starter_obtained_state.py can't reproduce the exact same
# Squirtle every run (see its module docstring) -- the hidden stats (IVs)
# depend on RNG state at creation time. Generating several fresh starters
# and reading their actual stats (rather than trusting a formula from
# memory -- one guess was wrong by 1 point) gave the real range at level
# 5:
#
#   HP/MaxHP: 19-20   Attack: 10-11   Defense: 11-13
#   Speed: 9-10       Special: 10-11
#
# The enemy's stats, by contrast, were checked the same way and never
# varied at all (always 20 HP, 10/10/10/12) across every regeneration --
# trainer-owned Pokemon in Gen 1 have fixed IVs, unlike the player's own,
# so only the player's stats need randomizing here.

ADDR_BATTLE_MON_ATTACK = 0xD025
ADDR_BATTLE_MON_DEFENSE = 0xD027
ADDR_BATTLE_MON_SPEED = 0xD029
ADDR_BATTLE_MON_SPECIAL = 0xD02B

BATTLE_MON_STAT_RANGES = {
    "hp": (19, 20),
    "attack": (10, 11),
    "defense": (11, 13),
    "speed": (9, 10),
    "special": (10, 11),
}


def write_u16(pyboy, addr, value):
    pyboy.memory[addr] = (value >> 8) & 0xFF
    pyboy.memory[addr + 1] = value & 0xFF


def randomize_battle_mon_stats(pyboy, rng):
    """
    ONLY valid for a level 5 starter -- the ranges below are the real
    spread for a freshly-obtained Lv5 Squirtle and nothing else.
    Applying it to a levelled Pokemon silently *downgrades* it: a Lv10
    Squirtle on 32 HP gets reset to 19-20. That cost a full round of
    misleading measurements after create_leveled_state.py existed, so
    callers working with a levelled party must leave it off.

    Roll new stats for the player's battle Pokemon within the range an
    actual freshly-obtained level-5 Squirtle could have, then set current
    HP to the (possibly new) max HP. `rng` is any object with `.randint`
    (e.g. Python's `random` module, or a seeded `random.Random`).
    """

    max_hp = rng.randint(*BATTLE_MON_STAT_RANGES["hp"])
    write_u16(pyboy, ADDR_BATTLE_MON_MAX_HP, max_hp)
    write_u16(pyboy, ADDR_BATTLE_MON_HP, max_hp)
    write_u16(pyboy, ADDR_BATTLE_MON_ATTACK, rng.randint(*BATTLE_MON_STAT_RANGES["attack"]))
    write_u16(pyboy, ADDR_BATTLE_MON_DEFENSE, rng.randint(*BATTLE_MON_STAT_RANGES["defense"]))
    write_u16(pyboy, ADDR_BATTLE_MON_SPEED, rng.randint(*BATTLE_MON_STAT_RANGES["speed"]))
    write_u16(pyboy, ADDR_BATTLE_MON_SPECIAL, rng.randint(*BATTLE_MON_STAT_RANGES["special"]))
