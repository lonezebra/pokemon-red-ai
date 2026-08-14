import random

from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state, ROUTE_1_ENTRY_STATE_PATH, wild_encounter_state_path
from core.controls import walk_tile, advance_battle_dialogue
from core.memory import is_in_battle, get_battle_type, get_enemy_mon_species, get_enemy_mon_level
from core.screen import save_screenshot

# Route 1 in Red/Blue only has two possible wild encounters (Pidgey and
# Rattata), so 2 distinct species is the real target -- but this doesn't
# hardcode that assumption, it just keeps sampling encounters (reloading
# a fresh walk into the grass each time, same as the training env's own
# reset()) until it's found this many *distinct* enemy species or gives
# up. Verified once by hand already (see core/memory.py's comment on
# ADDR_ENEMY_MON_SPECIES): a first capture like this is what confirmed
# 36 = Pidgey against the actual on-screen text.
TARGET_DISTINCT_SPECIES = 2
MAX_ATTEMPTS = 30
MAX_WALK_STEPS = 200

DIRECTIONS = ["up", "left", "right", "down"]


def find_one_encounter(pyboy, rng):
    """
    Walk randomly from a fresh Route 1 entry until a wild battle starts,
    then advance to the first FIGHT/PKMN/ITEM/RUN menu -- the same
    "known, natural start-of-episode point" every other battle state in
    this project uses.
    """

    load_state(pyboy, ROUTE_1_ENTRY_STATE_PATH)
    run_frames(pyboy, 30)

    for _ in range(MAX_WALK_STEPS):
        walk_tile(pyboy, rng.choice(DIRECTIONS), verbose=False)
        run_frames(pyboy, 5)

        if is_in_battle(pyboy):
            if get_battle_type(pyboy) != 1:
                # Not expected on Route 1, but if it ever happens, this
                # isn't the wild encounter we're looking for.
                return None
            advance_battle_dialogue(pyboy)
            return True

    return None


def main():
    pyboy = create_emulator()
    rng = random.Random(0)

    found_species = {}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if len(found_species) >= TARGET_DISTINCT_SPECIES:
            break

        result = find_one_encounter(pyboy, rng)
        if not result:
            print(f"Attempt {attempt}: no encounter found in {MAX_WALK_STEPS} steps, retrying.")
            continue

        species = get_enemy_mon_species(pyboy)
        level = get_enemy_mon_level(pyboy)

        if species in found_species:
            print(f"Attempt {attempt}: encountered species {species} again (already captured), skipping.")
            continue

        path = wild_encounter_state_path(species)
        save_state(pyboy, path)

        screenshot_name = f"wild_encounter_species{species}.png"
        save_screenshot(pyboy, screenshot_name)

        found_species[species] = level
        print(f"Attempt {attempt}: captured species {species} (level {level}) -> {path}")

    pyboy.stop()

    print()
    print(f"Captured {len(found_species)} distinct species: {found_species}")
    print("Check the saved screenshots to confirm each shows the expected Pokemon and level.")


if __name__ == "__main__":
    main()
