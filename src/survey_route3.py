import os

# Headless before any core import -- see train_forest_agent.py.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from core.config import PROJECT_ROOT  # noqa: E402
from core.controls import press_button, walk_tile, wait_for_free_movement  # noqa: E402
from core.emulator import run_frames  # noqa: E402
from core.memory import (  # noqa: E402
    get_party_hp,
    get_party_hp_fraction,
    get_party_max_hp,
    get_player_position,
)
from core.pathfind import walk_to_map, walk_to_tile  # noqa: E402
from build_map_panorama import build  # noqa: E402
from survey_viridian_forest import build_worker_handle_battle  # noqa: E402

# Survey Route 3, the corridor between Pewter City and Mt. Moon.
#
# Reuses the forest survey's battle handler unchanged (it was already
# map-agnostic: it compares the post-fight map against wherever the
# fight started, so a blackout aborts loudly instead of being recorded
# as topology). What can't be reused is the heal trip: the forest healed
# at Viridian's Pokemon Center via the Pallet-to-forest overworld chain,
# and Route 3 sits on the far side of the forest entirely. Its own
# nearest Center is Pewter's (map 58, confirmed by walking into it, not
# assumed from a map-ID table), and every leg of the round trip is a
# single adjacent-map hop -- Route 3 borders Pewter City directly -- so
# the chain machinery isn't needed at all.
#
# Party context this survey starts from: one Lv13-14 Squirtle. A live
# probe of the first trainer measured a win that still cost 15 HP
# (27 -> 12 of 38-41), so with ~8 trainers on the route, HP management
# is not optional. HEAL_BELOW_FRACTION matches the forest survey's
# 0.85: heal trips are cheap here (short hops, no maze), so healing
# aggressively costs little and every fight starts near full HP --
# the exact condition the trainer DQN's 100/100 evaluation measured.
# The trainers themselves are also the levelling plan: that same probe
# fight paid Lv13 -> Lv14, so by the route's end the party should be
# comfortably past its Lv14 ceiling without any separate grind.

PEWTER_CITY_MAP_ID = 2
PEWTER_POKECENTER_MAP_ID = 58
ROUTE_3_MAP_ID = 14

HEAL_BELOW_FRACTION = 0.85
MAX_HEAL_PRESSES = 40
# Route 3 is a corridor, not a maze; walks never legitimately need the
# forest's 10000-tile search budget.
TRAVEL_MAX_TILES = 3000


def heal_at_pewter_center(pyboy, handle_battle=None):
    """
    Walk to Pewter's Pokemon Center and talk to the nurse until HP is
    actually full -- checking real HP rather than counting presses, the
    same pattern as create_leveled_state.heal_at_pokemon_center, which
    this mirrors rather than reuses because that one hardcodes
    Viridian's Center and the overworld chain south of the forest.
    """
    for target in (PEWTER_CITY_MAP_ID, PEWTER_POKECENTER_MAP_ID):
        if get_player_position(pyboy)["map_id"] == target:
            continue
        if not walk_to_map(pyboy, target, handle_battle=handle_battle,
                            max_tiles=TRAVEL_MAX_TILES):
            return False

    # All Gen 1 Centers share one interior: door at the bottom, nurse
    # behind the counter at the top. Walk up until the counter blocks,
    # then talk.
    for _ in range(8):
        if not walk_tile(pyboy, "up", verbose=False):
            break
        run_frames(pyboy, 6)

    for _ in range(MAX_HEAL_PRESSES):
        if get_party_hp(pyboy) >= get_party_max_hp(pyboy):
            break
        press_button(pyboy, "a", hold_frames=12, release_frames=26)
        run_frames(pyboy, 20)

    if get_party_hp(pyboy) < get_party_max_hp(pyboy):
        return False

    # HP reads full mid-animation, before the nurse's trailing "OK.
    # We'll need your POKEMON." (or the following "Thank you...") box
    # has actually been cleared -- stopping on the HP read alone left
    # the player frozen behind a live text box that every direction then
    # reads as blocked, the same class of bug clear_overworld_dialogue
    # fixed for a beaten forest trainer's parting line. Confirmed live:
    # walk_to_map(2) failing right after a "successful" heal, and a
    # screenshot at that exact point showing the dialogue box still up.
    # wait_for_free_movement presses A until a real step succeeds, then
    # undoes that probe step, so it can't leave the player elsewhere.
    return wait_for_free_movement(pyboy)


def make_heal_if_needed(handle_battle):
    """
    Same contract as the forest survey's: called once per tile the
    survey is about to explore from; heals and walks back to that exact
    tile on success, and on any failure just returns False -- the caller
    restores its own pre-heal snapshot regardless, so failure costs an
    unhealed exploration of one tile, never correctness.
    """

    def heal_if_needed(pyboy, key):
        if get_party_hp_fraction(pyboy) >= HEAL_BELOW_FRACTION:
            return False

        x, y = key
        print(
            f"  HP {get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} "
            f"({get_party_hp_fraction(pyboy):.0%}) at {key} -- healing at Pewter"
        )

        if not heal_at_pewter_center(pyboy, handle_battle=handle_battle):
            print(f"    could not reach Pewter's Center from {key}; continuing unhealed")
            return False

        # walk_to_map only ever searches doorways on the map the player
        # is currently standing on (create_leveled_state.travel_to hit
        # this same lesson) -- healing leaves the party inside the
        # Center (map 58), which has no direct doorway to Route 3, so
        # asking for ROUTE_3_MAP_ID from there can never succeed. Step
        # out to Pewter City first, then to Route 3, matching the two
        # real doorways actually being crossed.
        if not walk_to_map(pyboy, PEWTER_CITY_MAP_ID, handle_battle=handle_battle,
                            max_tiles=TRAVEL_MAX_TILES):
            print("    healed, but could not get back to Pewter City; "
                  "continuing from the pre-heal snapshot")
            return False

        if not walk_to_map(pyboy, ROUTE_3_MAP_ID, handle_battle=handle_battle,
                            max_tiles=TRAVEL_MAX_TILES):
            print("    healed, but could not get back to Route 3; "
                  "continuing from the pre-heal snapshot")
            return False

        if not walk_to_tile(pyboy, x, y, stay_on_map=True, handle_battle=handle_battle,
                             max_tiles=TRAVEL_MAX_TILES):
            print(f"    healed and back on Route 3, but could not reach {key}; "
                  f"continuing from the pre-heal snapshot")
            return False

        print(f"    back at {key}, HP {get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} (healed)")
        return True

    return heal_if_needed


def build_worker_heal_if_needed(handle_battle):
    """Top-level factory, importable by reference from spawned workers --
    see core/parallel_survey.py for why this can't be a closure."""
    return make_heal_if_needed(handle_battle)


def main():
    build(
        "route3_entry", "route3",
        build_handle_battle=build_worker_handle_battle,
        build_heal_if_needed=build_worker_heal_if_needed,
    )


if __name__ == "__main__":
    main()
