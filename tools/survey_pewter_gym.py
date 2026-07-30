"""
Survey the Pewter Gym interior, fighting whatever stands in the way.

    cd src && ../.venv/bin/python3 ../tools/survey_pewter_gym.py

Produces map54_map.png and map54_map_meta.json from pewter_gym_entry.state.

Battle handling is on, via the same map-agnostic factory the forest survey
uses (it loads the trainer DQN fresh in each worker; make_handle_battle
compares before/after map IDs rather than hardcoding the forest). The Gym
is the first map since the battle-classification fixes where the survey
itself will genuinely fight: a Jr Trainer with a sight line across the
room, and Brock, whose battle starts by talking -- which is exactly what
the survey's trainer probe does to blocked tiles.

Losing to Brock here is an acceptable and informative outcome, not a
failure: a blackout lands the player in the Pokemon Center, the survey
records it as a map exit and restores its snapshot, and we learn -- from a
real fight rather than type-chart reasoning -- whether the current level-10
party and trainer DQN can plausibly take him. That answer decides whether
the Brock battle environment needs grinding/moveset work first.

No heal_if_needed: the room is a few dozen tiles with at most two fights,
against the forest's 713 tiles and six trainers.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from build_map_panorama import build
from survey_viridian_forest import build_worker_handle_battle


def main():
    build("pewter_gym_entry", "map54",
          build_handle_battle=build_worker_handle_battle)


if __name__ == "__main__":
    main()
