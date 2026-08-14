"""
Check the memory readers the whole-game agent's reward depends on against
save states whose contents this project already knows.

This exists because those readers are new and one of them -- get_badges --
reads an address that had never been read from code here at all, only cited
in prose. A reward function built on a wrong address doesn't crash, it just
trains toward nothing, which is exactly the failure mode this project has
paid for before (see the Route 22 section of README.md). So the rule is the
same as everywhere else: check against known ground truth first, then build
on it.

    cd src && ../.venv/bin/python3 ../tools/verify_whole_game_readers.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.config import PROJECT_ROOT  # noqa: E402
from core.emulator import create_emulator, run_frames  # noqa: E402
from core.memory import (  # noqa: E402
    get_badges,
    get_event_flags_sum,
    get_party_count,
    get_party_hp_total,
    get_party_hp_total_fraction,
    get_party_levels,
    get_party_max_hp_total,
    get_player_position,
)
from core.state import load_state  # noqa: E402

SAVES = PROJECT_ROOT / "saves"


def read_all(pyboy, state_path):
    load_state(pyboy, state_path)
    run_frames(pyboy, 10)
    return {
        "badges": get_badges(pyboy),
        "party_count": get_party_count(pyboy),
        "levels": get_party_levels(pyboy),
        "hp": get_party_hp_total(pyboy),
        "max_hp": get_party_max_hp_total(pyboy),
        "hp_fraction": get_party_hp_total_fraction(pyboy),
        "events": get_event_flags_sum(pyboy),
        "position": get_player_position(pyboy),
    }


# (state file, human description, list of (label, predicate on the reading)).
# Deliberately expressed as ranges and relationships rather than exact
# numbers wherever the exact number isn't independently known -- an
# assertion invented to match whatever the code currently prints would
# verify nothing at all.
CHECKS = [
    (
        "bedroom.state",
        "the very start -- no badges, no Pokemon",
        [
            ("no badges", lambda r: r["badges"] == 0),
            ("empty party", lambda r: r["party_count"] == 0),
            ("no levels listed", lambda r: r["levels"] == []),
            ("empty party reads 0.0 health, not a crash",
             lambda r: r["hp_fraction"] == 0.0),
        ],
    ),
    (
        "starter_obtained.state",
        "one starter, still badgeless",
        [
            ("no badges", lambda r: r["badges"] == 0),
            ("one Pokemon", lambda r: r["party_count"] == 1),
            ("one level, and it's the Lv5 starter",
             lambda r: len(r["levels"]) == 1 and r["levels"][0] == 5),
            ("at full health", lambda r: r["hp_fraction"] == 1.0),
        ],
    ),
    (
        "route3_leveled.state",
        "the levelled party Route 3 trains from",
        [
            ("still badgeless is wrong here -- Route 3 needs Boulder",
             lambda r: r["badges"] >= 1),
            ("party is levelled well past the Lv5 starter",
             lambda r: max(r["levels"]) >= 10),
            ("max HP grew with the level",
             lambda r: r["max_hp"] > 20),
        ],
    ),
    (
        "boulder_badge.state",
        "immediately after beating Brock",
        [
            ("exactly one badge", lambda r: r["badges"] == 1),
        ],
    ),
]

# Story progress must never go *backwards* along this chain, and must go
# meaningfully forwards end to end. This is the real test of the event-flag
# block: any single absolute count is unfalsifiable on its own, but the
# ordering is something the addresses either capture or don't.
#
# Not strictly-increasing at every step, on purpose. This first ran with that
# stricter rule and route2_entry.state "failed" it by matching
# pokedex_obtained.state exactly -- which is the correct reading, not a bad
# address: getting from Viridian to Route 2 is pure walking, and walking sets
# no story flags. An agent crossing a route earns its reward from the
# exploration term, not this one.
PROGRESSION = [
    "bedroom.state",
    "starter_obtained.state",
    "route_1_entry.state",
    "pokedex_obtained.state",
    "route2_entry.state",
    "boulder_badge.state",
]


def main():
    pyboy = create_emulator()
    failures = []
    readings = {}

    try:
        print("=" * 68)
        print("Per-state checks")
        print("=" * 68)

        for filename, description, assertions in CHECKS:
            path = SAVES / filename
            if not path.exists():
                print(f"\n{filename}: MISSING -- skipped")
                failures.append(f"{filename} does not exist")
                continue

            reading = read_all(pyboy, path)
            readings[filename] = reading

            print(f"\n{filename} ({description})")
            print(f"  badges={reading['badges']} "
                  f"party={reading['party_count']} "
                  f"levels={reading['levels']} "
                  f"hp={reading['hp']}/{reading['max_hp']} "
                  f"events={reading['events']} "
                  f"map={reading['position']['map_id']}")

            for label, predicate in assertions:
                ok = predicate(reading)
                print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
                if not ok:
                    failures.append(f"{filename}: {label}")

        print()
        print("=" * 68)
        print("Event flags must increase with story progress")
        print("=" * 68)
        print()

        previous_count = None
        previous_name = None
        first_count = None
        last_count = None
        for filename in PROGRESSION:
            path = SAVES / filename
            if not path.exists():
                print(f"{filename}: MISSING -- skipped")
                continue

            reading = readings.get(filename) or read_all(pyboy, path)
            count = reading["events"]

            if previous_count is None:
                first_count = count
                print(f"{filename:<28} events={count}")
            else:
                ok = count >= previous_count
                note = "" if count > previous_count else "  (unchanged -- navigation only)"
                print(f"{filename:<28} events={count} "
                      f"[{'PASS' if ok else 'FAIL'}] >= {previous_name}{note}")
                if not ok:
                    failures.append(
                        f"event flags went backwards from {previous_name} "
                        f"({previous_count}) to {filename} ({count})"
                    )

            previous_count, previous_name = count, filename
            last_count = count

        if first_count is not None and last_count is not None:
            ok = last_count > first_count
            print()
            print(f"[{'PASS' if ok else 'FAIL'}] end to end: "
                  f"{first_count} -> {last_count} events")
            if not ok:
                failures.append(
                    "event flags did not increase at all across the whole chain"
                )

    finally:
        pyboy.stop()

    print()
    print("=" * 68)
    if failures:
        print(f"{len(failures)} FAILURE(S) -- readers are not trustworthy yet")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 68)
        return 1

    print("All checks passed -- readers verified against known states")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
