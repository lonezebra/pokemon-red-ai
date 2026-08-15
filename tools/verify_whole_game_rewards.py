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
    MILESTONE_DELIVERED_REWARD,
    MILESTONE_PICKUP_REWARD,
    NEW_MAP_REWARD,
    NEW_TILE_REWARD,
    calculate_whole_game_reward,
)

MILESTONE_ITEM = 70  # OAKS_PARCEL_ITEM_ID -- see core/memory.py for how
                      # this value was verified

OAKS_LAB_MAP_ID = 40       # verified in core/memory's map-id ground truth
VIRIDIAN_CITY_MAP_ID = 1
ROUTE_1_MAP_ID = 12
ROUTE_22_MAP_ID = 33


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
        "worth real ground (>= 50 tiles) but a full tour stays under one delivery",
        NEW_TILE_REWARD * 50 <= NEW_MAP_REWARD
        and NEW_MAP_REWARD * 10 < MILESTONE_PICKUP_REWARD + MILESTONE_DELIVERED_REWARD,
        f"{NEW_MAP_REWARD} vs tile {NEW_TILE_REWARD}, "
        f"10-map tour {NEW_MAP_REWARD * 10} vs errand "
        f"{MILESTONE_PICKUP_REWARD + MILESTONE_DELIVERED_REWARD}",
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

    print("\nCarrying the Parcel is free (carry_home is retired)")
    # carry_home once paid +/-4 per hop toward/away from Oak while holding
    # the Parcel. Combined with the frontier bonus it taught the policy to
    # never pick the Parcel up at all (0/28 deliveries from a 36/36
    # checkpoint, the Mart never entered) -- holding had become a tax on the
    # touring the frontier bonus paid for. These checks are the regression
    # guard against any direction-dependent holding cost coming back.
    holding_at_viridian = state(
        milestone_items={MILESTONE_ITEM}, map_id=VIRIDIAN_CITY_MAP_ID
    )
    holding_at_route1 = state(
        milestone_items={MILESTONE_ITEM}, map_id=ROUTE_1_MAP_ID
    )
    holding_at_route22 = state(
        milestone_items={MILESTONE_ITEM}, map_id=ROUTE_22_MAP_ID
    )

    walks = [
        ("toward Oak", holding_at_viridian, holding_at_route1),
        ("away from Oak", holding_at_route1, holding_at_viridian),
        ("off the old map table entirely", holding_at_viridian, holding_at_route22),
    ]
    for label, src, dst in walks:
        total, comps = calculate_whole_game_reward(src, dst, False)
        results.append(check(
            f"walking {label} while holding it is direction-free",
            "carry_home" not in comps and total == comps["step"],
            f"total={total}",
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
