from stable_baselines3 import DQN

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT
from core.pathfind import _step
from core.memory import get_player_position
from core.battle_runner import fight_current_battle
from core.controls import wait_for_free_movement
from actions import get_action_name, num_actions
from rewards.forest_rewards import (
    calculate_forest_reward,
    position_key,
    FOREST_MAP_ID,
    CONNECTOR_MAP_ID,
)

FOREST_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "leveled.state"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"


class PokemonRedForestEnv:
    """
    Navigation through Viridian Forest toward its real exit (the
    connector room, map 47), established by an exhaustive parallel
    survey whose frontier was fully exhausted -- see rewards/
    forest_rewards.py for how that became a shortest-path-based reward
    instead of the straight-line -y potential Route 1/Route 2 used,
    since this map is a genuine maze (~45% of its own bounding box)
    rather than a corridor.

    Unlike every navigation env before this one, the forest has
    trainers: Gen 1 doesn't allow fleeing them, so encountering one is a
    forced fight. That fight is resolved transparently here by the
    already-solved trainer-battle DQN (100/100 in evaluation across all
    six known trainers) -- this env's action space never has any say
    over battle moves, only which direction to walk. Reusing
    pathfind._step directly (rather than reimplementing wild-encounter
    fleeing and trainer detection) is deliberate: that function is
    already the proven mechanism the forest survey itself depends on,
    including the detail that walking toward a trainer's tile does not
    start their battle by itself -- only pressing A does, and their
    pre-battle line has to be cleared first.

    No mid-episode healing, matching Route 1/Route 2's fixed-budget
    convention -- HP only ever goes down over an episode. Losing a
    forced battle isn't treated as a special case: a blackout lands the
    player on a Pokemon Center's map, which the reward function already
    penalizes as leaving the forest, and `done` already covers landing
    anywhere other than the forest or the connector room.
    """

    def __init__(self, max_steps=1000):
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_positions = set()
        self.trainer_model = DQN.load(str(TRAINER_MODEL_PATH))

    def _handle_battle(self, pyboy):
        fight_current_battle(pyboy, self.trainer_model)
        # Clears any trailing "you got $X" / trainer's parting line, win
        # or lose, the same as every other scripted interaction here.
        wait_for_free_movement(pyboy)

    def reset(self):
        load_state(self.pyboy, FOREST_ENTRY_STATE_PATH)
        run_frames(self.pyboy, 30)

        self.step_count = 0
        self.visited_positions = set()

        position = get_player_position(self.pyboy)
        self.visited_positions.add(position_key(position))

        return self._get_observation()

    def step(self, action_id):
        if action_id < 0 or action_id >= num_actions():
            raise ValueError(f"Invalid action_id: {action_id}")

        before = get_player_position(self.pyboy)
        direction = get_action_name(action_id)

        moved = _step(self.pyboy, direction, handle_battle=self._handle_battle)

        after = get_player_position(self.pyboy)
        reward = calculate_forest_reward(before=before, after=after)

        self.visited_positions.add(position_key(after))
        self.step_count += 1

        reached_goal = after["map_id"] == CONNECTOR_MAP_ID
        left_forest = after["map_id"] not in (FOREST_MAP_ID, CONNECTOR_MAP_ID)
        ran_out_of_steps = self.step_count >= self.max_steps

        done = reached_goal or left_forest or ran_out_of_steps

        info = {
            "moved": moved,
            "direction": direction,
            "before": before,
            "after": after,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return get_player_position(self.pyboy)

    def close(self):
        self.pyboy.stop()
