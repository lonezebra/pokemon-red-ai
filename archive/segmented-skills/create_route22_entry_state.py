import random

from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from core.state import save_state
from core.controls import walk_tile, attempt_run_from_wild_battle, wait_for_position_to_settle
from core.memory import get_player_position, print_player_position, is_in_battle
from core.screen import save_screenshot
from core.emulator import run_frames

# Reaches Route 22 -- the route WEST of Viridian City -- from the trained
# Route 1 policy's arrival point.
#
# **This script was originally, and wrongly, called
# create_route2_entry_state.py.** The mistake is worth recording, because
# it cost a full 1500-episode training run: a left-and-up-biased walk out
# of Viridian reaches map 33, and a screenshot confirmed that map is real
# outdoor route terrain (grass, not a building interior) -- so it was
# labelled Route 2 and an entire navigation task was built on it. What
# the screenshot actually confirmed was that map 33 is *a* route, never
# *which* route. Checking "is it outdoors?" felt like verification but
# didn't test the claim being made.
#
# Map 33 is Route 22, established three independent ways:
#   - Geometry: exiting map 33 eastward lands in Viridian City at x=0,
#     its far *west* edge -- so map 33 lies west of Viridian. Route 2 is
#     due north.
#   - Map numbering: a BFS flood-fill of Viridian City (see below) found
#     its four building doors are maps 41-44, which sits exactly where
#     the sequential map table puts Viridian's Pokecenter/Mart/School/
#     House given the two anchors this project had already verified
#     independently -- map 40 = Oak's Lab, map 12 = Route 1. The same
#     table puts Route 22 at 33.
#   - Behaviour: Route 22 dead-ends at the Victory Road gate, which
#     checks for all eight badges. Its west end is a solid mountain wall
#     (screenshotted). There is no forward exit this early in the game,
#     which is why ~1500 training episodes produced no real successes --
#     the task had no reachable goal, so no amount of reward shaping
#     would ever have fixed it.
#
# Kept (rather than deleted) because getting here took real probing and
# Route 22 matters much later, for Victory Road. It is a dead end for now.
#
# Separately, and the reason Route 2 is not reachable yet: a BFS
# flood-fill of Viridian City -- testing all four directions from every
# reachable tile via save states, until the frontier was exhausted --
# found the complete reachable set is 500 tiles whose ONLY exits are
# Route 1 (south), Route 22 (west), and those four building doors. There
# is no north exit at all. The northern frontier (x=6-13, y=4) is solid
# trees, and the path north around (19, 9) is blocked by an NPC saying
# "You can't go through here! This is private property!". Reaching the
# real Route 2 means opening that gate first -- see
# create_pokedex_obtained_state.py.

ROUTE_22_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "route22_entry.state"
VIRIDIAN_WALK_SEED = 50
VIRIDIAN_WALK_DIRECTIONS = ["up"] * 7 + ["left"] * 3
VIRIDIAN_WALK_MAX_STEPS = 100


def walk_route1_to_viridian(env, agent):
    obs = env.reset()
    info = {}
    for _ in range(300):
        action = agent.choose_action(obs, greedy=True)
        obs, reward, done, info = env.step(action)
        if done:
            break
    return info.get("reached_goal", False)


def walk_viridian_to_route22(pyboy):
    rng = random.Random(VIRIDIAN_WALK_SEED)

    for _ in range(VIRIDIAN_WALK_MAX_STEPS):
        walk_tile(pyboy, rng.choice(VIRIDIAN_WALK_DIRECTIONS), verbose=False)
        run_frames_battle_safe(pyboy)

        pos = get_player_position(pyboy)
        if pos["map_id"] != 1:
            return True

    return False


def run_frames_battle_safe(pyboy):
    run_frames(pyboy, 5)
    if is_in_battle(pyboy):
        attempt_run_from_wild_battle(pyboy)


def main():
    env = PokemonRedRoute1Env(max_steps=300)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(PROJECT_ROOT / "models" / "route1_q_table.json")

    print("Playing the trained Route 1 policy to Viridian City...")
    reached_viridian = walk_route1_to_viridian(env, agent)
    if not reached_viridian:
        print("Warning: did not reach Viridian City -- aborting.")
        env.close()
        return

    print_player_position(env.pyboy, "Arrived in Viridian City")

    print()
    print("Walking from Viridian City's entrance to Route 22...")
    reached_route22 = walk_viridian_to_route22(env.pyboy)
    if not reached_route22:
        print("Warning: never left Viridian City -- aborting.")
        env.close()
        return

    wait_for_position_to_settle(env.pyboy)
    print_player_position(env.pyboy, "Arrived at Route 22")
    save_screenshot(env.pyboy, "route22_entry.png")

    save_state(env.pyboy, ROUTE_22_ENTRY_STATE_PATH)
    print(f"Saved {ROUTE_22_ENTRY_STATE_PATH}")

    env.close()


if __name__ == "__main__":
    main()
