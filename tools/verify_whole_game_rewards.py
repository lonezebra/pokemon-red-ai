"""
Check the whole-game reward function scores events rather than conditions.

This exists because of a specific bug that reached real rollouts: the
blackout penalty was charged for *being* at zero HP rather than for going to
zero, and since a fainted party stays fainted for the hundreds of steps the
game spends walking you to a Pokemon Center, one blackout billed -5168
instead of -8 and buried every other term in the episode.

That is the third time this project has shipped the same mistake in a
different costume (leave_house_rewards.py's downstairs bonus, route1's
per-episode novelty term, now this), so it gets a test rather than a comment.
No emulator needed -- these are pure functions over two state dicts.

    ../.venv/bin/python3 tools/verify_whole_game_rewards.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewards.whole_game_rewards import (  # noqa: E402
    BADGE_REWARD,
    BLACKOUT_PENALTY,
    CARRY_HOME_SHAPING_PER_HOP,
    MAP_HOP_DISTANCE_TO_OAKS_LAB,
    MILESTONE_DELIVERED_REWARD,
    MILESTONE_PICKUP_REWARD,
    NEW_MAP_REWARD,
    NEW_TILE_REWARD,
    OAKS_LAB_MAP_ID,
    UNKNOWN_HOP_DISTANCE,
    calculate_whole_game_reward,
)

MILESTONE_ITEM = 70  # OAKS_PARCEL_ITEM_ID -- see core/memory.py for how
                      # this value was verified

VIRIDIAN_CITY_MAP_ID = 1   # hop 3 from Oak's Lab
ROUTE_1_MAP_ID = 12        # hop 2 from Oak's Lab
ROUTE_22_MAP_ID = 33       # hop 4 from Oak's Lab


def state(badges=0, events=0, levels=(5,), hp_fraction=1.0, blacked_out=False,
          milestone_items=frozenset(), map_id=OAKS_LAB_MAP_ID):
    return {
        "badges": badges,
        "events": events,
        "levels": list(levels),
        "hp_fraction": hp_fraction,
        "party_count": len(levels),
        "blacked_out": blacked_out,
        "milestone_items": frozenset(milestone_items),
        "map_id": map_id,
    }


def check(label, condition, detail=""):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f" -- {detail}" if detail else ""))
    return bool(condition)


def main():
    results = []

    print("Blackout is charged once, not per step")
    fainted = state(hp_fraction=0.0, blacked_out=True)

    _, entering = calculate_whole_game_reward(
        state(), fainted, tile_is_new=False
    )
    results.append(check(
        "the step the party goes down is penalised",
        entering["blackout"] == BLACKOUT_PENALTY,
        f"{entering['blackout']}",
    ))

    _, staying = calculate_whole_game_reward(
        fainted, fainted, tile_is_new=False
    )
    results.append(check(
        "staying fainted afterwards is free",
        staying["blackout"] == 0.0,
        f"{staying['blackout']}",
    ))

    # The real shape of the bug: one faint followed by a long walk to a
    # Pokemon Center. Anything worse than a single penalty means it is
    # being billed per step again.
    total = 0.0
    _, first = calculate_whole_game_reward(state(), fainted, False)
    total += first["blackout"]
    for _ in range(500):
        _, later = calculate_whole_game_reward(fainted, fainted, False)
        total += later["blackout"]
    results.append(check(
        "one faint plus 500 steps of recovery costs one penalty",
        total == BLACKOUT_PENALTY,
        f"{total} (would have been {BLACKOUT_PENALTY * 501} as a state check)",
    ))

    print("\nProgress is scored on the delta, never the level")
    two_badges = state(badges=2)
    _, held = calculate_whole_game_reward(two_badges, two_badges, False)
    results.append(check(
        "holding badges pays nothing per step",
        held["badge"] == 0.0,
        f"{held['badge']}",
    ))

    _, earned = calculate_whole_game_reward(
        state(badges=1), state(badges=2), False
    )
    results.append(check(
        "earning one badge pays once",
        earned["badge"] == BADGE_REWARD,
        f"{earned['badge']}",
    ))

    _, regressed = calculate_whole_game_reward(
        state(events=10), state(events=4), False
    )
    results.append(check(
        "an event counter going backwards is not a reward",
        regressed["event"] == 0.0,
        f"{regressed['event']}",
    ))

    print("\nHealing cannot be farmed")
    _, healed = calculate_whole_game_reward(
        state(hp_fraction=0.1), state(hp_fraction=1.0), False
    )
    _, half_healed = calculate_whole_game_reward(
        state(hp_fraction=0.5), state(hp_fraction=1.0), False
    )
    results.append(check(
        "a full heal is capped, not proportional without limit",
        healed["heal"] == half_healed["heal"],
        f"full={healed['heal']}, half={half_healed['heal']}",
    ))

    _, damaged = calculate_whole_game_reward(
        state(hp_fraction=1.0), state(hp_fraction=0.3), False
    )
    results.append(check(
        "taking damage is not a heal",
        damaged["heal"] == 0.0,
    ))

    print("\nExploration")
    _, fresh = calculate_whole_game_reward(state(), state(), tile_is_new=True)
    _, stale = calculate_whole_game_reward(state(), state(), tile_is_new=False)
    results.append(check(
        "a new tile beats a revisited one",
        fresh["explore"] > stale["explore"],
        f"{fresh['explore']} vs {stale['explore']}",
    ))

    print("\nThe frontier (entering a map this episode hasn't seen)")
    _, new_map = calculate_whole_game_reward(
        state(), state(), tile_is_new=True, map_is_new=True,
    )
    _, old_map = calculate_whole_game_reward(
        state(), state(), tile_is_new=True, map_is_new=False,
    )
    results.append(check(
        "a first-visit map pays the frontier bonus once",
        new_map["new_map"] == NEW_MAP_REWARD,
        f"{new_map['new_map']}",
    ))
    results.append(check(
        "re-entering a map this episode pays nothing",
        old_map["new_map"] == 0.0,
        f"{old_map['new_map']}",
    ))
    results.append(check(
        "the env's flag is the only trigger -- the default is off",
        calculate_whole_game_reward(state(), state(), True)[1]["new_map"] == 0.0,
    ))
    results.append(check(
        "worth real ground (>= 100 tiles) but less than a badge",
        NEW_TILE_REWARD * 100 <= NEW_MAP_REWARD < BADGE_REWARD,
        f"{NEW_MAP_REWARD} vs tile {NEW_TILE_REWARD}, badge {BADGE_REWARD}",
    ))

    print("\nMilestone items (Oak's Parcel, and anything added the same way)")
    empty_handed = state()
    holding = state(milestone_items={MILESTONE_ITEM})

    _, picked_up = calculate_whole_game_reward(empty_handed, holding, False)
    results.append(check(
        "picking one up pays once",
        picked_up["milestone"] == MILESTONE_PICKUP_REWARD,
        f"{picked_up['milestone']}",
    ))

    _, still_holding = calculate_whole_game_reward(holding, holding, False)
    results.append(check(
        "holding it for a subsequent step pays nothing",
        still_holding["milestone"] == 0.0,
        f"{still_holding['milestone']}",
    ))

    _, gave_it_away = calculate_whole_game_reward(holding, empty_handed, False)
    results.append(check(
        "delivering it pays once, worth more than the pickup",
        gave_it_away["milestone"] == MILESTONE_DELIVERED_REWARD
        and MILESTONE_DELIVERED_REWARD > MILESTONE_PICKUP_REWARD,
        f"{gave_it_away['milestone']}",
    ))

    _, never_had_it = calculate_whole_game_reward(empty_handed, empty_handed, False)
    results.append(check(
        "never having one at all is not mistaken for delivering it",
        never_had_it["milestone"] == 0.0,
        f"{never_had_it['milestone']}",
    ))

    print("\nCarrying the Parcel home")
    holding_at_viridian = state(
        milestone_items={MILESTONE_ITEM}, map_id=VIRIDIAN_CITY_MAP_ID
    )
    holding_at_route1 = state(
        milestone_items={MILESTONE_ITEM}, map_id=ROUTE_1_MAP_ID
    )
    holding_at_route22 = state(
        milestone_items={MILESTONE_ITEM}, map_id=ROUTE_22_MAP_ID
    )
    empty_at_viridian = state(map_id=VIRIDIAN_CITY_MAP_ID)
    empty_at_route1 = state(map_id=ROUTE_1_MAP_ID)

    _, closer = calculate_whole_game_reward(
        holding_at_viridian, holding_at_route1, False
    )
    results.append(check(
        "moving a hop closer while holding it pays one hop's worth",
        closer["carry_home"] == CARRY_HOME_SHAPING_PER_HOP,
        f"{closer['carry_home']}",
    ))

    _, farther = calculate_whole_game_reward(
        holding_at_route1, holding_at_viridian, False
    )
    results.append(check(
        "moving a hop farther while holding it costs the same amount",
        farther["carry_home"] == -CARRY_HOME_SHAPING_PER_HOP,
        f"{farther['carry_home']}",
    ))

    _, empty_handed_walk = calculate_whole_game_reward(
        empty_at_viridian, empty_at_route1, False
    )
    results.append(check(
        "the same walk without the Parcel pays nothing",
        empty_handed_walk["carry_home"] == 0.0,
        f"{empty_handed_walk['carry_home']}",
    ))

    _, pickup_step = calculate_whole_game_reward(
        empty_at_viridian, holding_at_viridian, False
    )
    results.append(check(
        "the pickup step itself isn't double-counted by carry_home",
        pickup_step["carry_home"] == 0.0,
        f"{pickup_step['carry_home']}",
    ))

    delivered_at_oaks_lab = state(map_id=OAKS_LAB_MAP_ID)
    holding_at_oaks_lab = state(
        milestone_items={MILESTONE_ITEM}, map_id=OAKS_LAB_MAP_ID
    )
    _, delivery_step = calculate_whole_game_reward(
        holding_at_oaks_lab, delivered_at_oaks_lab, False
    )
    results.append(check(
        "the delivery step itself isn't double-counted by carry_home",
        delivery_step["carry_home"] == 0.0,
        f"{delivery_step['carry_home']}",
    ))

    _, round_trip_out = calculate_whole_game_reward(
        holding_at_viridian, holding_at_route22, False
    )
    _, round_trip_back = calculate_whole_game_reward(
        holding_at_route22, holding_at_viridian, False
    )
    results.append(check(
        "a round trip through unmapped-adjacent ground nets zero, not farmable",
        round_trip_out["carry_home"] + round_trip_back["carry_home"] == 0.0,
        f"out={round_trip_out['carry_home']}, back={round_trip_back['carry_home']}",
    ))

    unmapped_map_id = 9999
    holding_somewhere_unmapped = state(
        milestone_items={MILESTONE_ITEM}, map_id=unmapped_map_id
    )
    _, into_unknown = calculate_whole_game_reward(
        holding_at_viridian, holding_somewhere_unmapped, False
    )
    results.append(check(
        "wandering off this table's edge while holding it is treated as moving away",
        into_unknown["carry_home"] < 0.0
        and unmapped_map_id not in MAP_HOP_DISTANCE_TO_OAKS_LAB
        and UNKNOWN_HOP_DISTANCE > max(MAP_HOP_DISTANCE_TO_OAKS_LAB.values()),
        f"{into_unknown['carry_home']}",
    ))

    print("\nStanding still")
    idle_total, _ = calculate_whole_game_reward(state(), state(), False)
    results.append(check(
        "doing nothing is never positive",
        idle_total <= 0.0,
        f"{idle_total}",
    ))

    print()
    failed = sum(1 for r in results if not r)
    if failed:
        print(f"{failed} FAILURE(S)")
        return 1
    print(f"All {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
