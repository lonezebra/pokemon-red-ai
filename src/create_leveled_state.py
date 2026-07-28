import random

from stable_baselines3 import DQN

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state
from core.config import PROJECT_ROOT
from core.controls import walk_tile, press_button, wait_for_free_movement
from core.pathfind import walk_to_map
from core.battle_runner import fight_current_battle
from core.memory import (
    get_player_position,
    get_party_level,
    get_party_hp,
    get_party_max_hp,
    get_party_hp_fraction,
    is_in_battle,
    get_battle_type,
    print_player_position,
)

# Levels the party up by fighting wild Pokemon with the trained
# wild-battle policy, then saves the result as a checkpoint.
#
# Why this exists. The trainer battles guarding Viridian Forest were
# measured, before any of this, to be unwinnable at the project's usual
# 90% bar: an "always attack" policy -- which is essentially optimal,
# since a Lv6 Squirtle's only damaging move is Tackle -- tops out around
# 67%, and the trained agent managed 30%. The blocker was never the
# policy. It was that the character is underlevelled for the fight, and
# a Lv6 Squirtle facing multi-Pokemon Bug Catcher parties loses to
# poison chip damage regardless of how well it picks moves.
#
# The in-game answer to being underlevelled is to go and train, and this
# project already has the piece needed: a wild-encounter policy that
# wins 100/100. So rather than tune a reward against an impossible
# target, this uses one solved skill to unblock another. Squirtle learns
# Bubble at Lv8, a second damaging move, which is the main thing worth
# reaching.
#
# Everything here is scaffolding in the README's sense -- it decides
# *where to walk* and *when to heal*, but every actual battle is played
# by the learned policy, not by a scripted "always use move 1".

WILD_MODEL_PATH = PROJECT_ROOT / "models" / "wild_battle_dqn.zip"
# Starts from the post-Parcel checkpoint, not route_1_entry.state.
# That matters: route_1_entry.state predates Oak's Parcel errand, so
# in it Viridian City's north gate is still shut and the finished,
# levelled save could never reach the forest it is meant for. Caught
# by exactly that failure -- the grind itself worked, then the last
# leg could not find Route 2.
START_STATE_PATH = PROJECT_ROOT / "saves" / "pokedex_obtained.state"
LEVELED_STATE_PATH = PROJECT_ROOT / "saves" / "leveled.state"

VIRIDIAN_FOREST_MAP_ID = 51
VIRIDIAN_CITY_MAP_ID = 1
VIRIDIAN_POKECENTER_MAP_ID = 41
ROUTE_2_MAP_ID = 13
FOREST_SOUTH_GATE_MAP_ID = 50
ROUTE_1_MAP_ID = 12

# Where the grinding happens. Route 1 rather than the forest, for one
# practical reason: it borders Viridian City, so a trip to the Pokemon
# Center and back is two short hops along ground this project has
# crossed reliably many times. Grinding in the forest meant healing
# via Route 2, and that southbound leg kept failing mid-trip, which
# stalled the whole run. Route 1's wild Pokemon are also exactly the
# Pidgey and Rattata the wild-battle policy was trained on, so it is
# fighting in-distribution rather than against unfamiliar bugs.
GRIND_MAP_ID = ROUTE_1_MAP_ID

TARGET_LEVEL = 10          # comfortably past Bubble at Lv8
HEAL_BELOW_FRACTION = 0.45  # heal before fainting rather than after
MAX_WALK_STEPS = 4000
MAX_HEAL_PRESSES = 40

DIRECTIONS = ("up", "down", "left", "right")


# The overworld maps between Pallet Town and Viridian Forest, in order.
# Travelling is done by walking this chain one adjacent map at a time,
# because pathfind.walk_to_map only ever looks for a doorway on the map
# it is standing on -- asking it for somewhere several maps away simply
# fails.
#
# Getting this wrong is what broke the first version: the forest's entry
# tile (17, 47) is *itself* one of its exit tiles, so a random walk
# stepping south immediately fell back into the gate. Recovery then
# asked for Viridian City while standing in the gate, which is not
# adjacent to it, and failed forever.
OVERWORLD_CHAIN = [
    0,   # Pallet Town
    12,  # Route 1
    1,   # Viridian City
    13,  # Route 2
    50,  # Viridian Forest south gate
    51,  # Viridian Forest
]

# Interiors, and which overworld map their door opens onto.
BUILDING_PARENT = {VIRIDIAN_POKECENTER_MAP_ID: VIRIDIAN_CITY_MAP_ID}


def travel_to(pyboy, target_map):
    """
    Walk to `target_map` from wherever the player currently is, including
    from inside a building -- which matters because fainting blacks the
    player out to a Pokemon Center rather than leaving them where they
    fell.
    """

    wait_for_free_movement(pyboy)

    current = get_player_position(pyboy)["map_id"]
    if current == target_map:
        return True

    if target_map in BUILDING_PARENT:
        if not travel_to(pyboy, BUILDING_PARENT[target_map]):
            return False
        return walk_to_map(pyboy, target_map)

    if current not in OVERWORLD_CHAIN:
        # Inside somewhere -- step out to whichever overworld map adjoins.
        for candidate in OVERWORLD_CHAIN:
            if walk_to_map(pyboy, candidate):
                break
        current = get_player_position(pyboy)["map_id"]
        if current not in OVERWORLD_CHAIN:
            print(f"  travel: stranded on map {current}")
            return False

    here = OVERWORLD_CHAIN.index(current)
    there = OVERWORLD_CHAIN.index(target_map)
    step = 1 if there > here else -1

    for index in range(here + step, there + step, step):
        if not walk_to_map(pyboy, OVERWORLD_CHAIN[index]):
            print(f"  travel: could not reach map {OVERWORLD_CHAIN[index]}")
            return False
    return True


def heal_at_pokemon_center(pyboy):
    """
    Talk to the nurse until HP is actually full again -- checking real HP
    rather than counting dialogue presses, the same pattern every other
    scripted interaction here uses.
    """

    if not travel_to(pyboy, VIRIDIAN_POKECENTER_MAP_ID):
        return False

    for _ in range(8):
        if not walk_tile(pyboy, "up", verbose=False):
            break
        run_frames(pyboy, 6)

    for _ in range(MAX_HEAL_PRESSES):
        if get_party_hp(pyboy) >= get_party_max_hp(pyboy):
            return True
        press_button(pyboy, "a", hold_frames=12, release_frames=26)
        run_frames(pyboy, 20)

    return get_party_hp(pyboy) >= get_party_max_hp(pyboy)


def return_to_forest(pyboy):
    if not travel_to(pyboy, VIRIDIAN_FOREST_MAP_ID):
        return False
    # The entry tiles are exits; step clear of them so an unlucky walk
    # south does not immediately leave again.
    for _ in range(4):
        if not walk_tile(pyboy, "up", verbose=False):
            break
        run_frames(pyboy, 6)
    return True


def grind(pyboy, model, target_level=TARGET_LEVEL, seed=0):
    rng = random.Random(seed)
    battles = 0

    for _ in range(MAX_WALK_STEPS):
        if get_party_level(pyboy) >= target_level:
            return battles

        # Fainting whites the player out to a Pokemon Center, which
        # heals but also teleports -- so top up before it can happen,
        # and treat being off the forest map as needing to walk back
        # either way.
        if get_party_hp_fraction(pyboy) < HEAL_BELOW_FRACTION:
            print(f"  HP {get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} -- healing")
            if not heal_at_pokemon_center(pyboy):
                print("  heal failed -- stopping")
                return battles
            travel_to(pyboy, GRIND_MAP_ID)
            continue

        if get_player_position(pyboy)["map_id"] != GRIND_MAP_ID:
            if not travel_to(pyboy, GRIND_MAP_ID):
                print("  could not get back to the grinding route -- stopping")
                return battles
            continue

        walk_tile(pyboy, rng.choice(DIRECTIONS), verbose=False)
        run_frames(pyboy, 5)

        if not is_in_battle(pyboy):
            continue

        if get_battle_type(pyboy) == 2:
            # A trainer -- not what we came for, and it cannot be fled.
            # Fight it anyway; losing just costs a trip to heal.
            print("  (trainer battle triggered)")

        fight_current_battle(pyboy, model)
        wait_for_free_movement(pyboy)
        battles += 1

        if battles % 3 == 0:
            print(f"  {battles} battles: Lv{get_party_level(pyboy)} "
                  f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")

    return battles


def main():
    model = DQN.load(str(WILD_MODEL_PATH))
    pyboy = create_emulator()
    load_state(pyboy, START_STATE_PATH)
    run_frames(pyboy, 30)

    wait_for_free_movement(pyboy)
    if not travel_to(pyboy, GRIND_MAP_ID):
        print("Could not reach the grinding route -- aborting.")
        pyboy.stop()
        return

    print(f"Start: Lv{get_party_level(pyboy)} "
          f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")
    print(f"Grinding wild battles up to Lv{TARGET_LEVEL}...")

    battles = grind(pyboy, model)

    print()
    print(f"Finished after {battles} battles: Lv{get_party_level(pyboy)} "
          f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")
    print_player_position(pyboy, "Ended at")

    print("Healing, then travelling to Viridian Forest for the checkpoint...")
    heal_at_pokemon_center(pyboy)
    if not travel_to(pyboy, VIRIDIAN_FOREST_MAP_ID):
        print("Warning: saved without reaching the forest.")

    save_state(pyboy, LEVELED_STATE_PATH)
    print(f"Saved {LEVELED_STATE_PATH}")

    pyboy.stop()


if __name__ == "__main__":
    main()
