import numpy as np

from core.emulator import run_frames
from core.controls import press_button, advance_battle_dialogue
from core.memory import (
    get_detailed_battle_state,
    get_move_cursor_slot,
    is_in_battle,
)

# Plays out a battle that is *already happening*, using a trained policy.
#
# The battle environments each own their own emulator and reset from a
# save state, which is right for training but useless for a live session
# that wandered into a fight on its own. This drives the same trained
# model against whatever battle the passed-in emulator is currently in,
# so scripted scaffolding (levelling up, the controller) can hand a real
# encounter to a learned policy instead of a hard-coded "always use move
# 1" fallback.

NUM_MOVE_SLOTS = 4
MAX_TURNS = 60


def battle_observation(state):
    """
    The same six numbers every battle environment in this project feeds
    its policy: own HP fraction, enemy HP fraction, and which move slots
    are actually usable.
    """

    return np.array(
        [
            state["battle_mon_hp"] / max(state["battle_mon_max_hp"], 1),
            state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1),
        ]
        + [
            1.0 if (state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0) else 0.0
            for i in range(NUM_MOVE_SLOTS)
        ],
        dtype=np.float32,
    )


def valid_move_slots(state):
    slots = [
        i
        for i in range(NUM_MOVE_SLOTS)
        if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
    ]
    return slots or [0]


def select_move(pyboy, slot_index):
    # Cursor-reading rather than a fixed press sequence: the move cursor
    # is sticky between turns and the list wraps, both found empirically
    # while building the rival battle environment.
    press_button(pyboy, "a", hold_frames=10, release_frames=15)

    current_slot = get_move_cursor_slot(pyboy)
    if current_slot is not None:
        for _ in range(max(0, slot_index - current_slot)):
            press_button(pyboy, "down", hold_frames=10, release_frames=15)
        for _ in range(max(0, current_slot - slot_index)):
            press_button(pyboy, "up", hold_frames=10, release_frames=15)

    press_button(pyboy, "a", hold_frames=10, release_frames=15)
    advance_battle_dialogue(pyboy)


def fight_current_battle(pyboy, model, max_turns=MAX_TURNS):
    """
    Fight until the battle ends. `model` is any Stable-Baselines3 policy
    taking the six-number observation above.

    A model trained with a run action (the wild-battle agent has one)
    may pick it here; since the point of calling this is usually to
    *win* rather than escape, and running is not even legal against a
    trainer, that choice is redirected to attacking. Same for naming a
    move slot that has no move or no PP left.
    """

    advance_battle_dialogue(pyboy)

    for _ in range(max_turns):
        if not is_in_battle(pyboy):
            return True

        state = get_detailed_battle_state(pyboy)
        slots = valid_move_slots(state)

        action, _ = model.predict(battle_observation(state), deterministic=True)
        action = int(action)
        if action not in slots:
            action = slots[0]

        select_move(pyboy, action)
        run_frames(pyboy, 5)

    return not is_in_battle(pyboy)
