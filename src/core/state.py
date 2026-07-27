from core.config import SAVE_DIR, EMULATION_SPEED


BEDROOM_STATE_PATH = SAVE_DIR / "bedroom.state"
STARTER_OBTAINED_STATE_PATH = SAVE_DIR / "starter_obtained.state"
RIVAL_BATTLE_STATE_PATH = SAVE_DIR / "rival_battle.state"
ROUTE_1_ENTRY_STATE_PATH = SAVE_DIR / "route_1_entry.state"

WILD_ENCOUNTER_STATE_DIR = SAVE_DIR / "wild_encounters"

# One state per distinct Viridian Forest trainer, each captured at the
# first FIGHT/PKMN/ITEM/RUN menu of that battle -- the same "start of
# episode" point every other battle state in this project uses.
TRAINER_BATTLE_STATE_DIR = SAVE_DIR / "trainer_battles"


def wild_encounter_state_path(species_id):
    # Named by Gen 1's internal species index (not the Pokedex number --
    # see core/memory.py's ADDR_ENEMY_MON_SPECIES) since that's what the
    # game and this project's own code actually key on; there can be more
    # than one of these (unlike every other state above), one per
    # distinct wild Pokemon species captured for training variety.
    return WILD_ENCOUNTER_STATE_DIR / f"species_{species_id}.state"


def load_bedroom_state(pyboy):
    if not BEDROOM_STATE_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {BEDROOM_STATE_PATH}. "
            "Create bedroom.state first."
        )

    with open(BEDROOM_STATE_PATH, "rb") as f:
        pyboy.load_state(f)

    pyboy.set_emulation_speed(EMULATION_SPEED)


def save_state(pyboy, path):
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pyboy.save_state(f)


def load_state(pyboy, path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find save state: {path}")

    with open(path, "rb") as f:
        pyboy.load_state(f)

    pyboy.set_emulation_speed(EMULATION_SPEED)