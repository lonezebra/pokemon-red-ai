"""
Hand-written controller: chains individually trained/scripted skills
together through one uniform interface, over a single continuous PyBoy
session, rather than each skill needing its own special-cased driving
code or its own separate emulator instance.

    bedroom.state -> [leave-house Q-agent]     -> Pallet Town
                   -> [scripted route]          -> Oak -> lab -> starter
                   -> [scripted route]          -> rival's trigger
                   -> [rival-battle DQN]        -> win/loss
                   -> [scripted route]          -> Route 1's entrance
                   -> [Route 1 Q-agent]         -> Viridian City

Further segments (Route 2, the forest, Pewter Gym, Route 3, ...) are
added the same way, one at a time, each verified before the next is
built on top of it -- see the project README's roadmap for what each of
those still needs.

Each learned skill (leave-house Q-agent, rival-battle DQN) is called
through agents.skills' choose_action(observation) -> action interface, so
this file never needs to know whether a given decision came from a
lookup table or a neural network. The scripted segments in between reuse
the exact same routes already verified in create_starter_obtained_state.py
and create_rival_battle_state.py -- this file just chains them onto one
live session instead of each reloading its own save state.

Getting from segment 1 to segment 2 needed one more fix beyond just
calling things in order: the leave-house Q-agent's episode ends the
instant map_id becomes 0 (Pallet Town), but that's not actually the
final resting position -- the game keeps auto-walking the player a
couple more tiles out of the doorway on its own afterward, with no input
needed. Trying to act immediately collided with that in-progress
movement and produced a nonsensical multi-tile position jump. Waiting
for position to stop changing on its own
(controls.wait_for_position_to_settle) fixed it, and reliably lands at
the same (5, 6) tile saves/outside_house.state represents -- exactly
where the scripted route to Oak's trigger already assumes it starts.

Getting from segment 3 to segment 4 needed the same treatment again:
winning the rival battle doesn't return control immediately (more
dialogue to clear first), and Oak's Lab's exit door is a different spot
in Pallet Town than the player's own house's door -- so it needed its
own wait_for_control_and_walk and wait_for_position_to_settle calls, and
its own scripted route (found by systematically probing which tiles
allowed movement) to the gap in the hedge that actually leads to Route 1.
See create_route1_entry_state.py's module docstring for the details.

Note: the individual Gymnasium environments (envs/simple_env.py,
envs/battle_env.py) each create and own their own emulator, which is
right for training (every episode needs a clean, fast reset) but not
directly reusable for "hand a live session from one skill to the next."
So the battle segment below re-implements battle_env.py's move-selection
and observation logic against the shared pyboy instance rather than
constructing a PokemonRedRivalBattleEnv -- a small amount of duplication,
traded for not reloading rival_battle.state partway through what's
supposed to be one continuous run.
"""

import numpy as np

from core.emulator import create_emulator, run_frames
from core.state import load_state, BEDROOM_STATE_PATH
from core.controls import (
    walk_tile,
    press_button,
    advance_battle_dialogue,
    wait_for_position_to_settle,
    attempt_run_from_wild_battle,
)
from core.memory import get_player_position, get_battle_state, get_move_cursor_slot, is_in_battle
from agents.skills import QTableSkill, RivalBattleSkill
from actions import get_action_name
from core.config import PROJECT_ROOT
from rewards.leave_house_rewards import PALLET_TOWN_MAP_ID
from rewards.route1_rewards import VIRIDIAN_CITY_MAP_ID
from create_starter_obtained_state import walk_to_oak_trigger, wait_for_lab_arrival, choose_starter
from create_rival_battle_state import walk_to_rival_trigger_and_battle
from create_route1_entry_state import walk_out_of_lab_and_up_to_route_1, ROUTE_1_MAP_ID


def run_leave_house_segment(pyboy, max_steps=200):
    print()
    print("Segment 1: bedroom.state -> leave-house Q-agent -> Pallet Town")
    print("-" * 62)

    skill = QTableSkill(PROJECT_ROOT / "models" / "leave_house_q_table.json")

    for step in range(max_steps):
        pos = get_player_position(pyboy)
        if pos["map_id"] == PALLET_TOWN_MAP_ID:
            break

        action = skill.choose_action(pos)
        walk_tile(pyboy, get_action_name(action), verbose=False)
        run_frames(pyboy, 10)

    if get_player_position(pyboy)["map_id"] != PALLET_TOWN_MAP_ID:
        print("Did not reach Pallet Town within the step limit.")
        return False

    # map_id becoming 0 isn't the final resting position -- see the
    # module docstring for why this matters.
    settled_pos = wait_for_position_to_settle(pyboy)
    print(f"Reached Pallet Town, settled at {settled_pos}.")
    return True


def run_starter_segment(pyboy, starter_name="squirtle"):
    print()
    print("Segment 2: Pallet Town -> Oak's trigger -> lab -> choose starter")
    print("-" * 62)

    print("Walking to Oak's trigger...")
    walk_to_oak_trigger(pyboy)

    print("Waiting for the automatic walk-in to the lab...")
    if not wait_for_lab_arrival(pyboy):
        print("Warning: did not reach the lab arrival tile as expected.")
        return False

    print(f"Choosing {starter_name}...")
    if not choose_starter(pyboy, starter_name):
        print("Warning: starter was not obtained.")
        return False

    print("Starter obtained.")
    return True


NUM_MOVE_SLOTS = 4


def _battle_observation(state):
    your_fraction = state["battle_mon_hp"] / max(state["battle_mon_max_hp"], 1)
    enemy_fraction = state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1)

    move_valid = [
        1.0 if (state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0) else 0.0
        for i in range(NUM_MOVE_SLOTS)
    ]

    return np.array([your_fraction, enemy_fraction] + move_valid, dtype=np.float32)


def _select_battle_move(pyboy, slot_index):
    # Same cursor-position-aware navigation as battle_env.py -- see that
    # file for why a fixed up/down sequence doesn't work here.
    press_button(pyboy, "a", hold_frames=10, release_frames=15)

    current_slot = get_move_cursor_slot(pyboy)
    if current_slot is not None:
        if slot_index > current_slot:
            for _ in range(slot_index - current_slot):
                press_button(pyboy, "down", hold_frames=10, release_frames=15)
        elif slot_index < current_slot:
            for _ in range(current_slot - slot_index):
                press_button(pyboy, "up", hold_frames=10, release_frames=15)

    press_button(pyboy, "a", hold_frames=10, release_frames=15)
    advance_battle_dialogue(pyboy)


def run_rival_battle_segment(pyboy, max_steps=30):
    print()
    print("Segment 3: walk to rival's trigger -> battle DQN -> win/loss")
    print("-" * 62)

    print("Walking to the rival's trigger and into the battle...")
    if not walk_to_rival_trigger_and_battle(pyboy):
        print("Warning: did not reach the battle as expected.")
        return False

    skill = RivalBattleSkill(PROJECT_ROOT / "models" / "rival_battle_dqn.zip")

    for _ in range(max_steps):
        state = get_battle_state(pyboy)
        if not state["in_battle"]:
            break

        valid_slots = [
            i
            for i in range(NUM_MOVE_SLOTS)
            if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
        ]

        action = skill.choose_action(_battle_observation(state))
        actual_action = action if action in valid_slots else valid_slots[0]
        _select_battle_move(pyboy, actual_action)

    final_state = get_battle_state(pyboy)
    won = final_state["enemy_mon_hp"] == 0

    print("Won the rival battle." if won else "Did not win the rival battle.")
    return won


def run_route1_entry_segment(pyboy):
    print()
    print("Segment 4: out of the lab -> Route 1's entrance")
    print("-" * 62)

    if not walk_out_of_lab_and_up_to_route_1(pyboy):
        print("Warning: did not reach Route 1 as expected.")
        return False

    pos = get_player_position(pyboy)
    print(f"Reached Route 1 (map {ROUTE_1_MAP_ID}) at {pos}.")
    return True


def run_route1_navigation_segment(pyboy, max_steps=150):
    print()
    print("Segment 5: Route 1 Q-agent -> Viridian City")
    print("-" * 62)

    skill = QTableSkill(PROJECT_ROOT / "models" / "route1_q_table.json")

    for _ in range(max_steps):
        pos = get_player_position(pyboy)
        if pos["map_id"] == VIRIDIAN_CITY_MAP_ID:
            break

        action = skill.choose_action(pos)
        walk_tile(pyboy, get_action_name(action), verbose=False)
        run_frames(pyboy, 10)

        # Route 1 is tall grass -- a step can trigger a wild encounter at
        # any point. Fleeing it transparently here matches route1_env.py's
        # own handling: navigation is the only thing this segment's skill
        # was trained to solve, so battle interruptions never reach it.
        if is_in_battle(pyboy):
            attempt_run_from_wild_battle(pyboy)

    if get_player_position(pyboy)["map_id"] != VIRIDIAN_CITY_MAP_ID:
        print("Did not reach Viridian City within the step limit.")
        return False

    print(f"Reached Viridian City at {get_player_position(pyboy)}.")
    return True


def main():
    print("Pokemon Red AI -- controller (one continuous run)")

    pyboy = create_emulator()
    load_state(pyboy, BEDROOM_STATE_PATH)
    run_frames(pyboy, 60)

    leave_house_ok = run_leave_house_segment(pyboy)
    starter_ok = run_starter_segment(pyboy) if leave_house_ok else False
    battle_ok = run_rival_battle_segment(pyboy) if starter_ok else False
    route1_entry_ok = run_route1_entry_segment(pyboy) if battle_ok else False
    route1_nav_ok = run_route1_navigation_segment(pyboy) if route1_entry_ok else False

    pyboy.stop()

    print()
    print("Summary")
    print("-" * 62)
    print(f"Leave-house segment:    {'OK' if leave_house_ok else 'FAILED'}")
    print(f"Starter segment:        {'OK' if starter_ok else 'FAILED'}")
    print(f"Rival-battle segment:   {'OK' if battle_ok else 'FAILED'}")
    print(f"Route 1 entry segment:  {'OK' if route1_entry_ok else 'FAILED'}")
    print(f"Route 1 nav segment:    {'OK' if route1_nav_ok else 'FAILED'}")


if __name__ == "__main__":
    main()
