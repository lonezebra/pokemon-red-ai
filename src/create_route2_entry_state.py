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

# Findings from probing this by hand, starting from the trained Route 1
# policy's arrival in Viridian City:
#
#   - Straight "up" from the Route 1/Viridian boundary runs into a
#     one-way ledge almost immediately (blocks upward movement, same
#     ledge mechanic Route 1 itself has) -- not a story gate or NPC, just
#     terrain, confirmed by walking directly into it and screenshotting
#     the result rather than assuming.
#   - Random exploration from the arrival point wanders into Viridian's
#     own buildings just as often as it finds the real way out (two
#     different building interiors turned up, map IDs 43 and 44, both
#     entered at the generic "just walked in the door" tile (2, 7) --
#     Viridian City has more than one shop/house near this path).
#   - A left-and-up-biased walk (fixed seed, for reproducibility) reaches
#     map 33 reliably at position (39, 6) -- confirmed by screenshot to
#     be real outdoor route terrain (grass, not a building interior),
#     i.e. Route 2.
#   - Route 2 itself turned out to be considerably more maze-like than
#     Route 1: several thousand steps of further scripted scouting from
#     this entry point (biased up, down, and left-biased attempts, plus
#     directly testing the one visible ledge) never found the actual
#     exit toward Viridian Forest. Rather than keep hand-scouting
#     indefinitely, this is where scripted scaffolding stops and the
#     training task (route2_env.py) takes over -- see
#     rewards/route2_rewards.py for how its reward function handles not
#     knowing the exact goal in advance.

ROUTE_2_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "route2_entry.state"
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


def walk_viridian_to_route2(pyboy):
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
    print("Walking from Viridian City's entrance to Route 2...")
    reached_route2 = walk_viridian_to_route2(env.pyboy)
    if not reached_route2:
        print("Warning: never left Viridian City -- aborting.")
        env.close()
        return

    wait_for_position_to_settle(env.pyboy)
    print_player_position(env.pyboy, "Arrived at Route 2")
    save_screenshot(env.pyboy, "route2_entry.png")

    save_state(env.pyboy, ROUTE_2_ENTRY_STATE_PATH)
    print(f"Saved {ROUTE_2_ENTRY_STATE_PATH}")

    env.close()


if __name__ == "__main__":
    main()
