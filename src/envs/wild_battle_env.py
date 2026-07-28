import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.emulator import create_emulator, run_frames
from core.state import load_state, wild_encounter_state_path, WILD_ENCOUNTER_STATE_DIR
from core.controls import press_button, advance_battle_dialogue
from core.memory import get_detailed_battle_state, get_move_cursor_slot, randomize_battle_mon_stats
from rewards.wild_battle_rewards import calculate_wild_battle_reward


NUM_MOVE_SLOTS = 4
RUN_ACTION = 4
NUM_ACTIONS = 5


class PokemonRedWildBattleEnv(gym.Env):
    """
    A Gymnasium environment for wild Pokemon encounters, resetting from
    one of the save states in saves/wild_encounters/ (see
    create_wild_encounter_state.py) picked at random each episode --
    unlike the fixed rival matchup, the opponent's species and level
    actually vary here, which is the entire reason this is a separate
    environment rather than a mode of PokemonRedRivalBattleEnv.

    Action: 0-3 pick a move slot (same invalid-slot handling as the rival
    battle env: an unusable slot costs a small penalty instead of
    pressing any button). Action 4 attempts to run -- always a legal
    choice, unlike a move slot, since Gen 1 always lets you at least try
    to flee a wild encounter (never a trainer battle, which is why this
    action doesn't exist on PokemonRedRivalBattleEnv). Running has a
    real chance to fail (Gen 1's flee mechanic is a speed comparison),
    in which case the wild Pokemon still gets its turn -- exactly like a
    failed attack, just without dealing any damage.

    Observation: [your_hp_fraction, enemy_hp_fraction,
                  move1_valid, move2_valid, move3_valid, move4_valid]
    Deliberately not including species/level directly for this first
    version -- same reasoning as leave-house/rival-battle starting
    minimal: HP fractions and move availability might already be enough
    signal to learn sensible fight-or-flee behavior, and this is cheap
    to revisit if evaluation says otherwise.

    randomize_stats=True (the default) rerolls the player's own battle
    stats each reset, same as the rival battle env and for the same
    reason (saves/wild_encounters/*.state each only capture one exact IV
    roll).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=30, randomize_stats=True):
        super().__init__()

        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.randomize_stats = randomize_stats
        self.step_count = 0

        self.encounter_paths = sorted(WILD_ENCOUNTER_STATE_DIR.glob("species_*.state"))
        if not self.encounter_paths:
            raise FileNotFoundError(
                f"No wild encounter states found in {WILD_ENCOUNTER_STATE_DIR}. "
                "Run create_wild_encounter_state.py first."
            )

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        encounter_path = random.choice(self.encounter_paths)
        load_state(self.pyboy, encounter_path)
        run_frames(self.pyboy, 10)

        if self.randomize_stats:
            randomize_battle_mon_stats(self.pyboy, random)

        self.step_count = 0

        state = get_detailed_battle_state(self.pyboy)
        return self._observation(state), {}

    def step(self, action):
        before = get_detailed_battle_state(self.pyboy)

        if action == RUN_ACTION:
            self._attempt_run()
            chose_invalid_action = False
        else:
            valid_slots = self._valid_move_slots(before)
            chose_invalid_action = action not in valid_slots
            actual_action = valid_slots[0] if chose_invalid_action else action
            self._select_move(actual_action)

        after = get_detailed_battle_state(self.pyboy)
        reward = calculate_wild_battle_reward(before, after, invalid_action=chose_invalid_action)

        self.step_count += 1

        terminated = not after["in_battle"]
        truncated = self.step_count >= self.max_steps

        info = {
            "before": before,
            "after": after,
            "chose_invalid_action": chose_invalid_action,
            "won": terminated and after["enemy_mon_hp"] == 0,
            "lost": terminated and after["battle_mon_hp"] == 0,
            "fled": terminated and after["enemy_mon_hp"] > 0 and after["battle_mon_hp"] > 0,
        }

        return self._observation(after), reward, terminated, truncated, info

    def _valid_move_slots(self, state):
        return [
            i
            for i in range(NUM_MOVE_SLOTS)
            if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
        ]

    def _select_move(self, slot_index):
        # Same cursor-reading approach as PokemonRedRivalBattleEnv --
        # see its docstring for why a fixed press sequence isn't
        # reliable (sticky, wrapping move cursor).
        press_button(self.pyboy, "a", hold_frames=10, release_frames=15)

        current_slot = get_move_cursor_slot(self.pyboy)
        if current_slot is not None:
            if slot_index > current_slot:
                for _ in range(slot_index - current_slot):
                    press_button(self.pyboy, "down", hold_frames=10, release_frames=15)
            elif slot_index < current_slot:
                for _ in range(current_slot - slot_index):
                    press_button(self.pyboy, "up", hold_frames=10, release_frames=15)

        press_button(self.pyboy, "a", hold_frames=10, release_frames=15)
        advance_battle_dialogue(self.pyboy)

    def _attempt_run(self):
        # The top FIGHT/PKMN/ITEM/RUN menu resets to FIGHT every turn
        # (unlike the sticky move-list cursor) -- from FIGHT: down moves
        # to ITEM, right moves to RUN. One attempt only, since each env
        # step is meant to be one real in-game turn -- if it fails, the
        # agent sees that in the next observation and can choose to run
        # again (or fight instead) on its own, rather than this silently
        # retrying on its behalf the way controls.attempt_run_from_wild_
        # battle() deliberately does for the scripted Route 1 fallback.
        press_button(self.pyboy, "down", hold_frames=10, release_frames=15)
        press_button(self.pyboy, "right", hold_frames=10, release_frames=15)
        press_button(self.pyboy, "a", hold_frames=10, release_frames=15)
        advance_battle_dialogue(self.pyboy)

    def _observation(self, state):
        your_fraction = state["battle_mon_hp"] / max(state["battle_mon_max_hp"], 1)
        enemy_fraction = state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1)

        move_valid = [
            1.0 if (state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0) else 0.0
            for i in range(NUM_MOVE_SLOTS)
        ]

        return np.array([your_fraction, enemy_fraction] + move_valid, dtype=np.float32)

    def close(self):
        self.pyboy.stop()
