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


def read_u16(pyboy, addr):
    """Read a big-endian 2-byte value."""
    return (pyboy.memory[addr] << 8) | pyboy.memory[addr + 1]


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
