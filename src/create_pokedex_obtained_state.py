from envs.route1_env import PokemonRedRoute1Env
from agents.q_learning_agent import QLearningAgent
from actions import num_actions
from core.config import PROJECT_ROOT
from core.emulator import run_frames
from core.state import save_state
from core.controls import walk_tile, press_button
from core.pathfind import walk_to_map
from core.memory import get_player_position, print_player_position, has_item, OAKS_PARCEL_ITEM_ID
from core.screen import save_screenshot
from create_starter_obtained_state import wait_for_control_and_walk

# Runs Oak's Parcel errand: Viridian Mart -> Oak's Lab -> Pokedex.
#
# Why this exists at all. Viridian City's northern exit -- the only way
# to Route 2, Viridian Forest and eventually Pewter City -- is closed
# until this errand is done, and this project's controller skipped it
# entirely (bedroom -> lab -> starter -> rival -> Route 1). That was
# invisible until a BFS flood-fill of Viridian showed its complete
# reachable set was 500 tiles with exits only to Route 1, Route 22, and
# four building doors. No north exit existed, so no navigation agent was
# ever going to find one.
#
# Measured directly, before and after running this script:
#
#            reachable tiles   y range   exits north
#   before        500           4 - 35   none
#   after         600           0 - 35   (17,0) (18,0) (19,0) -> map 13
#
# Map 13 is Route 2, arrived at (7-9, 71) -- its southern end, and a
# useful fact in itself: Route 2 runs to y=71, so like Route 1 it is a
# tall vertical corridor, which is exactly the shape the y-coordinate
# potential shaping in rewards/route1_rewards.py was built for.
#
# Everything here is scripted scaffolding, in the sense the README
# allows: it unlocks a milestone so a *learned* agent has somewhere to
# go, rather than teaching anything about playing.

POKEDEX_OBTAINED_STATE_PATH = PROJECT_ROOT / "saves" / "pokedex_obtained.state"

VIRIDIAN_CITY_MAP_ID = 1
VIRIDIAN_MART_MAP_ID = 42
ROUTE_1_MAP_ID = 12
PALLET_TOWN_MAP_ID = 0
OAKS_LAB_MAP_ID = 40

# The Mart clerk's script runs on its own the moment you step inside --
# it walks the player to the counter and hands over the Parcel without
# any route of our own. So rather than counting presses (unreliable, as
# every other dialogue in this project has shown), press A and watch the
# bag until the Parcel actually appears.
MAX_MART_PRESSES = 60

# Oak stands at the top of his lab; walk up until the Parcel leaves the
# bag, which is the delivery genuinely happening rather than a guess at
# how many tiles away he is.
MAX_LAB_APPROACH_STEPS = 16
PRESSES_PER_APPROACH_STEP = 6

# The post-delivery scene (Oak's reaction, the rival barging in, both
# Pokedexes) is long and its length isn't worth measuring precisely --
# wait_for_control_and_walk below stops the moment control really
# returns, so this only needs to be generous.
CUTSCENE_PRESSES = 140


def walk_route1_to_viridian(env, agent, max_steps=300):
    obs = env.reset()
    info = {}
    for _ in range(max_steps):
        action = agent.choose_action(obs, greedy=True)
        obs, _, done, info = env.step(action)
        if done:
            break
    return info.get("reached_goal", False)


def collect_parcel(pyboy):
    for _ in range(MAX_MART_PRESSES):
        press_button(pyboy, "a", hold_frames=10, release_frames=25)
        run_frames(pyboy, 12)
        if has_item(pyboy, OAKS_PARCEL_ITEM_ID):
            return True
    return False


def deliver_parcel(pyboy):
    for _ in range(MAX_LAB_APPROACH_STEPS):
        walk_tile(pyboy, "up", verbose=False)
        run_frames(pyboy, 10)
        for _ in range(PRESSES_PER_APPROACH_STEP):
            press_button(pyboy, "a", hold_frames=10, release_frames=25)
            run_frames(pyboy, 12)
            if not has_item(pyboy, OAKS_PARCEL_ITEM_ID):
                return True
    return False


def main():
    env = PokemonRedRoute1Env(max_steps=300)
    agent = QLearningAgent(num_actions=num_actions())
    agent.load(PROJECT_ROOT / "models" / "route1_q_table.json")
    pyboy = env.pyboy

    print("Playing the trained Route 1 policy to Viridian City...")
    if not walk_route1_to_viridian(env, agent):
        print("Warning: did not reach Viridian City -- aborting.")
        env.close()
        return
    print_player_position(pyboy, "Arrived in Viridian City")

    print("\nHeading into the Viridian Mart...")
    if not walk_to_map(pyboy, VIRIDIAN_MART_MAP_ID):
        print("Warning: could not find the Mart door -- aborting.")
        env.close()
        return

    print("Collecting Oak's Parcel...")
    if not collect_parcel(pyboy):
        print("Warning: the Parcel never arrived in the bag -- aborting.")
        env.close()
        return
    print("Got Oak's Parcel.")

    # Clearing the clerk's remaining dialogue and stepping out the door
    # are the same action: press A until a real move south succeeds.
    wait_for_control_and_walk(pyboy, "down")
    if get_player_position(pyboy)["map_id"] == VIRIDIAN_MART_MAP_ID:
        walk_to_map(pyboy, VIRIDIAN_CITY_MAP_ID)

    print("\nWalking back to Pallet Town...")
    for map_id, label in (
        (ROUTE_1_MAP_ID, "Route 1"),
        (PALLET_TOWN_MAP_ID, "Pallet Town"),
        (OAKS_LAB_MAP_ID, "Oak's Lab"),
    ):
        if not walk_to_map(pyboy, map_id):
            print(f"Warning: could not reach {label} -- aborting.")
            env.close()
            return
        print(f"  reached {label}: {get_player_position(pyboy)}")

    print("\nDelivering the Parcel to Oak...")
    if not deliver_parcel(pyboy):
        print("Warning: Oak never took the Parcel -- aborting.")
        env.close()
        return
    print("Parcel delivered.")

    for _ in range(CUTSCENE_PRESSES):
        press_button(pyboy, "a", hold_frames=10, release_frames=20)
    wait_for_control_and_walk(pyboy, "down")

    print_player_position(pyboy, "After the Pokedex scene")
    save_screenshot(pyboy, "pokedex_obtained.png")
    save_state(pyboy, POKEDEX_OBTAINED_STATE_PATH)
    print(f"Saved {POKEDEX_OBTAINED_STATE_PATH}")

    env.close()


if __name__ == "__main__":
    main()
