from core.emulator import create_emulator, run_frames
from core.state import load_state, ROUTE_1_ENTRY_STATE_PATH
from core.controls import walk_tile, attempt_run_from_wild_battle
from core.memory import get_player_position, is_in_battle
from actions import get_action_name, num_actions
from rewards.route1_rewards import (
    calculate_route1_reward,
    position_key,
    ROUTE_1_MAP_ID,
    VIRIDIAN_CITY_MAP_ID,
)


class PokemonRedRoute1Env:
    """
    Same shape as PokemonRedLeaveHouseEnv (envs/simple_env.py): not
    Gymnasium, just reset()/step() returning (observation, reward, done,
    info). Reused deliberately, since this is the same kind of task --
    walk from a known start to a known goal map -- just a longer route.

    The one real difference from the leave-house task: Route 1 is tall
    grass, so movement can be interrupted at any step by a wild Pokemon
    encounter. Since learning battle strategy is a separate, later
    milestone (and running is always legal against a wild Pokemon,
    unlike the rival's forced trainer battle), this environment handles
    that automatically -- if a step triggers a battle, it immediately
    tries to run from it before returning control to the agent, so the
    navigation task the agent actually has to learn stays just
    navigation, with the battle interruption a transparent, scripted
    detail rather than something it has to solve too.
    """

    def __init__(self, max_steps=150):
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_positions = set()
        self.encounters = 0

    def reset(self):
        load_state(self.pyboy, ROUTE_1_ENTRY_STATE_PATH)
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

        reward = calculate_route1_reward(
            before=before,
            after=after,
            visited_positions=self.visited_positions,
        )

        self.visited_positions.add(position_key(after))

        self.step_count += 1

        reached_goal = after["map_id"] == VIRIDIAN_CITY_MAP_ID
        left_route_1 = after["map_id"] not in (ROUTE_1_MAP_ID, VIRIDIAN_CITY_MAP_ID)
        ran_out_of_steps = self.step_count >= self.max_steps

        done = reached_goal or left_route_1 or ran_out_of_steps

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
