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
