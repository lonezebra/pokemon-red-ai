import functools

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
from create_leveled_state import heal_at_pokemon_center, travel_to
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
DIAGNOSTIC_STATE_PATH = PROJECT_ROOT / "saves" / "forest_survey_last_trainer.state"
HEAL_BELOW_FRACTION = 0.85
# pathfind.DEFAULT_MAX_TILES (1200) is well past what any other map here
# has needed, but build_map_panorama's own survey of this one is allowed
# up to 2500 -- a heal round trip starting deep inside the forest can
# legitimately need to search nearly the whole map to find its way back
# to the entrance, so its budget has to match or it fails long before
# the forest itself runs out of unexplored tiles. Confirmed: a heal
# attempted from (25, 18) failed to find a path out at the 1200 default.
#
# 2500 itself later proved insufficient too, for a different reason:
# unlike the outer survey_map call (which accretes one shared, cached
# `tiles`/`states` set across the whole run), every individual heal trip
# is its own fresh, cache-less walk_to_map/walk_to_tile search that has
# to rediscover a large fraction of the maze from scratch each time --
# so the deeper a heal is triggered from, the more of the map that one
# search has to re-explore blind. Confirmed: a heal from (1, 15) failed
# to find map 50 even at 5000.
TRAVEL_MAX_TILES = 10000


def make_handle_battle(model):
    def handle_battle(pyboy):
        position = get_player_position(pyboy)
        before_map = position["map_id"]
        before = get_detailed_battle_state(pyboy)
        print(
            f"  trainer at {(position['x'], position['y'])}: "
            f"our Lv{get_party_level(pyboy)} HP{get_party_hp(pyboy)}/"
            f"{get_party_max_hp(pyboy)} ({get_party_hp_fraction(pyboy):.0%}) "
            f"vs enemy species {before['enemy_mon_species']} "
            f"Lv{before['enemy_mon_level']}"
        )
        # Kept for reproduction if this fight is the one that loses --
        # the training states only ever cover the six trainers captured
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
        if after_position["map_id"] != before_map:
            # A loss blacks the party out to the last Pokemon Center,
            # which would otherwise look to the BFS like a legitimate
            # exit from this tile -- so the whole run is aborted rather
            # than silently mislabeling a loss as map topology. Compared
            # against whatever map the fight actually started on, not a
            # hardcoded constant, since this same handler now gets reused
            # to survey past the forest too (map 47, map 2, ...). At
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


def build_worker_handle_battle():
    """
    Top-level, zero-arg factory so a parallel_survey_map worker process
    can build its own handle_battle rather than one being shared across
    processes -- see core/parallel_survey.py for why. Loads its own copy
    of the model rather than inheriting whatever the driver has loaded,
    matching train_route1_agent_parallel.py's workers, which likewise
    each build their own agent from scratch inside the child process.
    """
    model = DQN.load(str(MODEL_PATH))
    return make_handle_battle(model)


def make_heal_if_needed(handle_battle, target_map, min_fraction=HEAL_BELOW_FRACTION):
    """
    Called once per tile the survey is about to explore from. If HP is
    below `min_fraction`, travels all the way out to the Viridian Pokemon
    Center, heals, and walks back to the exact tile (already known
    reachable, so this is a fresh but unsurprising walk_to_tile search,
    not a repeat of any actual fighting) before letting the survey
    continue -- `handle_battle` is threaded through the whole round trip
    in case a not-yet-discovered trainer sits somewhere along the way.

    `target_map` is whichever map is actually being surveyed -- this
    handler now gets reused past the forest itself (map 47, map 2, ...),
    so "return to the forest" can't be hardcoded the way it first was;
    the return leg is a plain travel_to(target_map) instead.

    Healing is best-effort, not mandatory. A heal attempted from (1, 15)
    failed to find map 50 identically at both a 2500 and a 10000 tile
    budget -- raising the cap made no difference at all, which points at
    a real one-way barrier (this part of Viridian Forest likely has a
    ledge, same as several routes in this game) rather than a search that
    just needed more room. Past a point like that, "heal by walking back"
    is not a recoverable situation no matter the budget, so any failure
    along the round trip is just abandoned rather than escalated: this
    function returns False and the caller (survey_map) restores its own
    saved snapshot for `key` regardless of what this function returns or
    where it leaves the emulator, so there is nothing to recover here --
    walking back to `key` only matters for capturing an updated, healed
    snapshot on success, never for correctness on failure.
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
            print(f"    could not reach the Pokemon Center from {key} -- likely "
                  f"a one-way barrier past this point; continuing unhealed")
            return False

        if not travel_to(pyboy, target_map, handle_battle=handle_battle, max_tiles=TRAVEL_MAX_TILES):
            print(f"    healed, but could not return to map {target_map} from the "
                  f"Pokemon Center; continuing from the pre-heal snapshot at {key}")
            return False

        if not walk_to_tile(pyboy, x, y, stay_on_map=True, handle_battle=handle_battle,
                             max_tiles=TRAVEL_MAX_TILES):
            print(f"    healed and returned to map {target_map}, but could not "
                  f"navigate back to {key}; continuing from the pre-heal snapshot instead")
            return False

        print(f"    back at {key}, HP{get_party_hp(pyboy)}/{get_party_max_hp(pyboy)} (healed)")
        return True

    return heal_if_needed


def build_worker_heal_if_needed(handle_battle, target_map):
    """
    heal_if_needed's own factory signature parallel_survey_map expects
    is (handle_battle) -> heal_if_needed, but which map to return to
    after healing depends on what's being surveyed and has to be decided
    before the survey starts. Bind target_map via functools.partial at
    the call site instead of a closure over it here -- a closure isn't
    picklable at all, and parallel_survey_map's workers are spawned (a
    real new interpreter re-importing everything, not forked via
    copy-on-write) specifically to avoid a fork-after-threading-init
    hazard PyTorch's own thread pool hit here once already, so anything
    handed to a worker has to survive real pickling now, not just
    fork's memory reuse.
    """
    return make_heal_if_needed(handle_battle, target_map)


FOREST_MAP_ID = 51


def main():
    # Runs across this machine's full core count by default (see
    # build_map_panorama.build's parallel=True) -- pass parallel=False
    # only when specifically watching one process at a time, e.g. while
    # debugging a change to handle_battle/heal_if_needed themselves.
    build(
        "leveled", "forest",
        build_handle_battle=build_worker_handle_battle,
        build_heal_if_needed=functools.partial(build_worker_heal_if_needed, target_map=FOREST_MAP_ID),
    )


if __name__ == "__main__":
    main()
