from stable_baselines3 import DQN

from core.config import PROJECT_ROOT
from core.state import save_state
from core.battle_runner import fight_current_battle
from core.controls import wait_for_free_movement
from core.pathfind import walk_to_tile
from core.memory import (
    get_player_position,
    get_detailed_battle_state,
    get_party_level,
    get_party_hp,
    get_party_hp_fraction,
    get_party_max_hp,
)
from create_leveled_state import heal_at_pokemon_center, return_to_forest
from build_map_panorama import build

# Re-runs the Viridian Forest survey now that its trainers are beatable.
#
# The first survey (see README, "Viridian Forest is surveyed but
# blocked") found only 676 reachable tiles and an exit list that led
# straight back to Route 2 -- because a trainer occupies its tile,
# survey_map()'s plain flood fill correctly reads one as a wall, the same
# as any other unwalkable tile. That result was real, but it was a
# statement about what's reachable *without* fighting, not a statement
# about the forest's actual layout. Now that a Lv10 party with the
# trainer-battle DQN evaluates at 100/100, the survey can be handed a
# battle handler that fights and wins instead of giving up -- so this
# finds the forest's true exit toward Pewter City, if it has one.
#
# The first attempt at that lost a fight and blacked out. Instrumenting
# it showed why: HP drifted from 100% down to 81% over an unhealed run of
# five-plus trainers in a row (one single fight alone cost 56% of max
# HP), and the loss came fighting a sixth already worn down. The 100/100
# evaluation only ever measured fresh, full-HP battles -- entering one
# already damaged is a situation the policy has never seen. Rather than
# trust it to generalize to that, HEAL_BELOW_FRACTION below makes the
# survey manage HP itself, the same way create_leveled_state.py's grind
# loop does for the levelling grind.

MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
FOREST_MAP_ID = 51
DIAGNOSTIC_STATE_PATH = PROJECT_ROOT / "saves" / "forest_survey_last_trainer.state"
HEAL_BELOW_FRACTION = 0.85
# pathfind.DEFAULT_MAX_TILES (1200) is well past what any other map here
# has needed, but build_map_panorama's own survey of this one is allowed
# up to 2500 -- a heal round trip starting deep inside the forest can
# legitimately need to search nearly the whole map to find its way back
# to the entrance, so its budget has to match or it fails long before
# the forest itself runs out of unexplored tiles. Confirmed: a heal
# attempted from (25, 18) failed to find a path out at the 1200 default.
TRAVEL_MAX_TILES = 2500


def make_handle_battle(model):
    def handle_battle(pyboy):
        position = get_player_position(pyboy)
        before = get_detailed_battle_state(pyboy)
        print(
            f"  trainer at {(position['x'], position['y'])}: "
            f"our Lv{get_party_level(pyboy)} HP{get_party_hp(pyboy)}/"
            f"{get_party_max_hp(pyboy)} ({get_party_hp_fraction(pyboy):.0%}) "
            f"vs enemy species {before['enemy_mon_species']} "
            f"Lv{before['enemy_mon_level']}"
        )
        # Kept for reproduction if this fight is the one that loses --
        # the training states only ever cover the five trainers captured
        # before this point, at full HP, so this is the only record of
        # what a fight further in (or fought while already worn down by
        # an earlier one, since nothing here heals between battles the
        # way create_leveled_state.py's grind loop does) actually looks
        # like.
        save_state(pyboy, DIAGNOSTIC_STATE_PATH)

        fight_current_battle(pyboy, model)
        # Clears any trailing "you got $X" / trainer's parting line --
        # the same trailing dialogue create_leveled_state.py's grind()
        # clears after every battle, win or lose.
        wait_for_free_movement(pyboy)

        after_position = get_player_position(pyboy)
        if after_position["map_id"] != FOREST_MAP_ID:
            # A loss blacks the party out to the last Pokemon Center,
            # which would otherwise look to the BFS like a legitimate
            # exit from this tile -- so the whole run is aborted rather
            # than silently mislabeling a loss as forest topology. At
            # 100/100 in evaluation this should not happen at full HP,
            # but that evaluation never fought two battles back to back
            # without healing in between, and this survey does.
            raise RuntimeError(
                f"Lost a trainer battle during the survey -- blacked out "
                f"to map {after_position['map_id']}. Went in at "
                f"HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} vs "
                f"enemy species {before['enemy_mon_species']} "
                f"Lv{before['enemy_mon_level']}. Diagnostic state saved to "
                f"{DIAGNOSTIC_STATE_PATH}."
            )
        print(f"    won, now HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")

    return handle_battle


def make_heal_if_needed(handle_battle, min_fraction=HEAL_BELOW_FRACTION):
    """
    Called once per tile the survey is about to explore from. If HP is
    below `min_fraction`, travels all the way out to the Viridian Pokemon
    Center, heals, and walks back to the exact tile (already known
    reachable, so this is a fresh but unsurprising walk_to_tile search,
    not a repeat of any actual fighting) before letting the survey
    continue -- `handle_battle` is threaded through the whole round trip
    in case a not-yet-discovered trainer sits somewhere along the way.
    """

    def heal_if_needed(pyboy, key):
        if get_party_hp_fraction(pyboy) >= min_fraction:
            return False

        x, y = key
        print(
            f"  HP {get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} "
            f"({get_party_hp_fraction(pyboy):.0%}) at {key} -- returning to heal"
        )

        if not heal_at_pokemon_center(pyboy, handle_battle=handle_battle, max_tiles=TRAVEL_MAX_TILES):
            raise RuntimeError(f"Could not reach the Pokemon Center to heal from {key}")
        if not return_to_forest(pyboy, handle_battle=handle_battle, max_tiles=TRAVEL_MAX_TILES):
            raise RuntimeError(f"Could not return to the forest after healing from {key}")
        if not walk_to_tile(pyboy, x, y, stay_on_map=True, handle_battle=handle_battle,
                             max_tiles=TRAVEL_MAX_TILES):
            raise RuntimeError(f"Could not navigate back to {key} after healing")

        print(f"    back at {key}, HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)}")
        return True

    return heal_if_needed


def main():
    model = DQN.load(str(MODEL_PATH))
    handle_battle = make_handle_battle(model)
    build(
        "leveled", "forest",
        handle_battle=handle_battle,
        heal_if_needed=make_heal_if_needed(handle_battle),
    )


if __name__ == "__main__":
    main()
