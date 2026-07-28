import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.emulator import create_emulator, run_frames
from core.state import load_state, TRAINER_BATTLE_STATE_DIR
from core.controls import press_button, advance_battle_dialogue
from core.memory import get_detailed_battle_state, get_move_cursor_slot, randomize_battle_mon_stats
from rewards.trainer_battle_rewards import calculate_trainer_battle_reward


NUM_MOVE_SLOTS = 4


class PokemonRedTrainerBattleEnv(gym.Env):
    """
    A Gymnasium environment for the trainer battles that block Viridian
    Forest, resetting from one of the save states in
    saves/trainer_battles/ (see create_trainer_battle_states.py) picked
    at random each episode.

    This exists because a survey of Viridian Forest found its only
    reachable exit leads back the way the agent came: the way onward is
    guarded by Bug Catchers, and a trainer occupies its tile, so getting
    past one means beating it. There is deliberately **no run action** --
    unlike PokemonRedWildBattleEnv, Gen 1 does not allow fleeing a
    trainer, so the only way out of this environment is to win or lose.

    Action: which move slot to use (0-3). An unusable slot (unknown move,
    or no PP left) costs a small penalty, and the environment substitutes
    the first valid move and plays it -- never leaving the state
    unchanged, which is the deadlock the rival battle env hit when an
    invalid pick made the observation identical forever.

    Observation: [your_hp_fraction, enemy_hp_fraction,
                  move1_valid, move2_valid, move3_valid, move4_valid]
    Same six numbers as the other battle environments. Notably this does
    not include the opponent's level even though the captured trainers
    range Lv6-Lv9 against our Lv6 -- starting minimal is the pattern that
    has worked so far, and if evaluation shows the agent losing
    specifically to the higher-level ones, that is the obvious first
    thing to add.

    randomize_stats defaults to **False** here, unlike the other battle
    environments. Their randomisation exists to stop a policy memorising
    one exact IV roll, but memory.randomize_battle_mon_stats only knows
    the stat range of a freshly-obtained *level 5* starter. These battles
    are fought by a deliberately levelled party (see
    create_leveled_state.py), so rerolling would quietly reset a Lv10
    Squirtle's 32 HP back to about 20 -- which is exactly what happened,
    and made a first round of Lv10 measurements meaningless. The cost is
    that each captured state carries one IV roll; worth revisiting with
    level-aware randomisation if a policy starts looking overfitted.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=60, randomize_stats=False):
        super().__init__()

        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.randomize_stats = randomize_stats
        self.step_count = 0

        self.battle_paths = sorted(TRAINER_BATTLE_STATE_DIR.glob("trainer_*.state"))
        if not self.battle_paths:
            raise FileNotFoundError(
                f"No trainer battle states found in {TRAINER_BATTLE_STATE_DIR}. "
                "Run create_trainer_battle_states.py first."
            )

        self.action_space = spaces.Discrete(NUM_MOVE_SLOTS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        load_state(self.pyboy, random.choice(self.battle_paths))
        run_frames(self.pyboy, 10)

        if self.randomize_stats:
            randomize_battle_mon_stats(self.pyboy, random)

        self.step_count = 0

        state = get_detailed_battle_state(self.pyboy)
        return self._observation(state), {}

    def step(self, action):
        before = get_detailed_battle_state(self.pyboy)

        valid_slots = self._valid_move_slots(before)
        chose_invalid_action = action not in valid_slots
        actual_action = valid_slots[0] if chose_invalid_action else action
        self._select_move(actual_action)

        after = get_detailed_battle_state(self.pyboy)
        reward = calculate_trainer_battle_reward(
            before, after, invalid_action=chose_invalid_action
        )

        self.step_count += 1

        terminated = not after["in_battle"]
        truncated = self.step_count >= self.max_steps

        info = {
            "before": before,
            "after": after,
            "chose_invalid_action": chose_invalid_action,
            "won": terminated and after["battle_mon_hp"] > 0,
            "lost": terminated and after["battle_mon_hp"] == 0,
            "knocked_one_out": (
                after["in_battle"]
                and after["enemy_mon_species"] != before["enemy_mon_species"]
            ),
        }

        return self._observation(after), reward, terminated, truncated, info

    def _valid_move_slots(self, state):
        slots = [
            i
            for i in range(NUM_MOVE_SLOTS)
            if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
        ]
        # Every move being out of PP is possible in a long fight; falling
        # back to slot 0 keeps the battle progressing (the game itself
        # falls back to Struggle) rather than raising mid-episode.
        return slots or [0]

    def _select_move(self, slot_index):
        # Cursor-reading rather than a fixed press sequence -- the move
        # cursor is sticky between turns and the list wraps around, both
        # found empirically while building the rival battle env.
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
