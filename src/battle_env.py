import numpy as np
import gymnasium as gym
from gymnasium import spaces

from emulator import create_emulator, run_frames
from state import load_state, RIVAL_BATTLE_STATE_PATH
from controls import press_button, advance_battle_dialogue
from memory import get_battle_state, get_move_cursor_slot
from battle_rewards import calculate_battle_reward


NUM_MOVE_SLOTS = 4


class PokemonRedRivalBattleEnv(gym.Env):
    """
    A Gymnasium environment for the first rival battle, reset from
    saves/rival_battle.state (the very first FIGHT/ITEM/RUN menu, before
    either side has moved).

    Action: which move slot to use (0-3). A slot with no known move, or
    0 PP remaining, is invalid -- picking one costs a small penalty and
    does not press any button (see battle_rewards.INVALID_MOVE_PENALTY).
    This is the "dynamic, PP-aware masking" from the project's design
    notes, implemented as a penalty rather than true action masking,
    since it only ever matters for this Pokemon's first couple of moves.

    Observation: [your_hp_fraction, enemy_hp_fraction,
                  move1_valid, move2_valid, move3_valid, move4_valid]
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps=30):
        super().__init__()

        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0

        self.action_space = spaces.Discrete(NUM_MOVE_SLOTS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        load_state(self.pyboy, RIVAL_BATTLE_STATE_PATH)
        run_frames(self.pyboy, 10)

        self.step_count = 0

        state = get_battle_state(self.pyboy)
        return self._observation(state), {}

    def step(self, action):
        before = get_battle_state(self.pyboy)
        valid_slots = self._valid_move_slots(before)

        if action not in valid_slots:
            after = before
            reward = calculate_battle_reward(before, after, invalid_action=True)
        else:
            self._select_move(action)
            after = get_battle_state(self.pyboy)
            reward = calculate_battle_reward(before, after)

        self.step_count += 1

        terminated = not after["in_battle"]
        truncated = self.step_count >= self.max_steps

        info = {
            "before": before,
            "after": after,
            "valid_slots": valid_slots,
        }

        return self._observation(after), reward, terminated, truncated, info

    def _valid_move_slots(self, state):
        return [
            i
            for i in range(NUM_MOVE_SLOTS)
            if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
        ]

    def _select_move(self, slot_index):
        # From the top FIGHT/ITEM/RUN menu: A selects FIGHT. The move
        # cursor is sticky (it remembers the last move used, it does not
        # reset each turn) and the list wraps around rather than clamping
        # at the ends -- both discovered empirically -- so instead of
        # guessing a fixed sequence of presses, read the cursor's actual
        # position and move it the exact number of steps in the right
        # direction.
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
