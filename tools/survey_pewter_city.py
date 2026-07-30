"""
Regenerate the Pewter City panorama and survey meta from the rebuilt
chain checkpoint.

    cd src && ../.venv/bin/python3 ../tools/survey_pewter_city.py

The chain rebuild (71e9d7b, c653203) restored the three checkpoint .state
files the container rollback destroyed, but not the map-2 panorama or its
meta -- the chain only surveys far enough to find each leg's exit tile,
and the standalone panorama was never re-run. This runs the standard
build() against pewter_city_entry.state, which produces both at once:
map2_map.png (the visual proof this really is Pewter City -- the labelled
GYM and MART) and map2_map_meta.json, whose exits list is how the Gym's
door tile is found for the interior survey that follows.

No battle handling: Pewter City's streets have no trainers, and the survey
treats building doors as exits rather than walking through them.

Guarded for spawn: the parallel survey re-imports __main__ in each worker.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from build_map_panorama import build


def main():
    build("pewter_city_entry", "map2")


if __name__ == "__main__":
    main()
