from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT
from core.controls import walk_tile, attempt_run_from_wild_battle
from core.memory import get_player_position, is_in_battle
from actions import get_action_name, num_actions
from rewards.route2_rewards import calculate_route2_reward, position_key, ROUTE_2_MAP_ID

ROUTE_2_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "route2_entry.state"


class PokemonRedRoute2Env:
    """
    Same shape as PokemonRedRoute1Env -- reset()/step() from a known
    start (saves/route2_entry.state, just past Viridian City's north
    exit), reward and battle-handling following the same pattern. The
    real difference: Route 1's goal (Viridian City) and general layout
    (a mostly-vertical corridor) were both known before training began.
    Route 2's actual path to Viridian Forest wasn't pinned down by
    scripted scouting (see route2_rewards.py's docstring), so "done" here
    is any map transition away from Route 2 at all, not one specific
    map ID -- successfully reaching some *other* new area, whatever it
    turns out to be, is what counts as reached_goal.
    """

    def __init__(self, max_steps=1000):
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_positions = set()
        self.encounters = 0

    def reset(self):
        load_state(self.pyboy, ROUTE_2_ENTRY_STATE_PATH)
        run_frames(self.pyboy, 30)

        self.step_count = 0
        self.visited_positions = set()
        self.encounters = 0

        position = get_player_position(self.pyboy)
        self.visited_positions.add(position_key(position))

        return self._get_observation()

    def step(self, action_id):
        if action_id < 0 or action_id >= num_actions():
            raise ValueError(f"Invalid action_id: {action_id}")

        before = get_player_position(self.pyboy)

        direction = get_action_name(action_id)
        moved = walk_tile(self.pyboy, direction, verbose=False)
        run_frames(self.pyboy, 10)

        if is_in_battle(self.pyboy):
            self.encounters += 1
            attempt_run_from_wild_battle(self.pyboy)

        after = get_player_position(self.pyboy)

        reward = calculate_route2_reward(
            before=before,
            after=after,
            visited_positions=self.visited_positions,
        )

        self.visited_positions.add(position_key(after))

        self.step_count += 1

        reached_goal = after["map_id"] not in (ROUTE_2_MAP_ID, 1)
        left_route_2 = after["map_id"] == 1
        ran_out_of_steps = self.step_count >= self.max_steps

        done = reached_goal or left_route_2 or ran_out_of_steps

        info = {
            "moved": moved,
            "direction": direction,
            "before": before,
            "after": after,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
            "encounters": self.encounters,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return get_player_position(self.pyboy)

    def close(self):
        self.pyboy.stop()
