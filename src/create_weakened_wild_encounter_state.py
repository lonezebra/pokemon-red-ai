import os

# Headless before any core import -- see train_forest_agent.py.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from core.emulator import create_emulator, run_frames  # noqa: E402
from core.state import load_state, save_state, WILD_ENCOUNTER_STATE_DIR  # noqa: E402
from core.memory import (  # noqa: E402
    get_detailed_battle_state,
    get_enemy_mon_species,
    is_in_battle,
)
from core.battle_runner import select_move  # noqa: E402

# Captures a second save state per existing wild encounter, with the
# enemy already fought down to low HP -- for mixing into training
# alongside the full-HP states, so the network sees many direct
# "catch from here" experiences instead of depending on random
# exploration to land in that window by chance.
#
# Why this exists: measured live, catching at this HP band succeeds
# 17/17 and 19/19 across the two species this project currently has
# wild-encounter states for -- essentially guaranteed. But average
# episode length under random exploration is only ~2.5 steps (these
# opponents are weak enough that a single hit often goes straight from
# mid-HP to zero), so naturally *landing* in the weakened-but-alive
# window is rare regardless of how many total training steps run --
# the same shape of problem the forest curriculum solved by starting
# some episodes directly at the hard part instead of hoping
# exploration finds it. A 200,000-timestep retrain still valued a
# guaranteed-catch state at ~3 out of a possible ~15, essentially
# unchanged from a 50,000-step run's ~2.3 -- more of the same lever
# wasn't working.
#
# Weakening always uses move slot 0, matching the exact sequence that
# produced the verified 100% catch rate live -- this only needs to
# reliably reach the target band, not showcase good play.
TARGET_HP_FRACTION = 0.12
MAX_TURNS = 15


def weaken_and_save(pyboy, source_path, output_path):
    load_state(pyboy, source_path)
    run_frames(pyboy, 10)

    for _ in range(MAX_TURNS):
        state = get_detailed_battle_state(pyboy)
        if not state["in_battle"]:
            return False, "enemy fainted before reaching the target HP band"

        fraction = state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1)
        if fraction <= TARGET_HP_FRACTION:
            break

        select_move(pyboy, 0)
        run_frames(pyboy, 10)
    else:
        return False, f"still above {TARGET_HP_FRACTION:.0%} after {MAX_TURNS} turns"

    state = get_detailed_battle_state(pyboy)
    if not state["in_battle"]:
        return False, "fainted on the final weakening turn"

    save_state(pyboy, output_path)
    return True, state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1)


def weakened_state_path(species_id):
    return WILD_ENCOUNTER_STATE_DIR / f"species_{species_id}_weakened.state"


def main():
    source_paths = sorted(WILD_ENCOUNTER_STATE_DIR.glob("species_*.state"))
    # Only ever operate on the full-HP captures, never re-weaken an
    # already-weakened file if this is run again after the pool grows.
    source_paths = [p for p in source_paths if "_weakened" not in p.name]

    if not source_paths:
        print(f"No wild encounter states found in {WILD_ENCOUNTER_STATE_DIR}. "
              "Run create_wild_encounter_state.py first.")
        return 1

    pyboy = create_emulator()
    failures = 0

    for path in source_paths:
        species = get_species_from_path(path)
        output_path = weakened_state_path(species)
        ok, detail = weaken_and_save(pyboy, path, output_path)
        if ok:
            print(f"species {species}: weakened to {detail:.1%} HP -> {output_path.name}")
        else:
            failures += 1
            print(f"species {species}: FAILED ({detail})")

    pyboy.stop()
    print(f"\n{len(source_paths) - failures}/{len(source_paths)} weakened states captured")
    return 0 if failures == 0 else 1


def get_species_from_path(path):
    # species_36.state -> 36. Reading it back from the actual captured
    # state rather than parsing the filename would also work, but the
    # filename is already the source of truth wild_encounter_state_path
    # writes, so parsing it keeps this in one line and one place.
    return int(path.stem.split("_")[1])


if __name__ == "__main__":
    import sys
    sys.exit(main())
