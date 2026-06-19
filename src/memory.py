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