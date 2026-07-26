from core.emulator import create_emulator, run_frames
from core.state import load_bedroom_state
from core.controls import walk_tile
from core.memory import get_player_position
from actions import get_action_name, num_actions
from rewards.leave_house_rewards import calculate_leave_house_reward, position_key, PALLET_TOWN_MAP_ID


class PokemonRedLeaveHouseEnv:
    """
    A tiny reinforcement-learning-style environment.

    This is intentionally simple and beginner-friendly.

    It is not Gymnasium yet.
    It just follows the same basic idea:

        observation = env.reset()
        observation, reward, done, info = env.step(action)
    """

    def __init__(self, max_steps=200):
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_positions = set()

    def reset(self):
        """
        Reset the environment to the saved bedroom state.
        """

        load_bedroom_state(self.pyboy)
        run_frames(self.pyboy, 60)

        self.step_count = 0
        self.visited_positions = set()

        position = get_player_position(self.pyboy)
        self.visited_positions.add(position_key(position))

        return self._get_observation()

    def step(self, action_id):
        """
        Apply one action and return:
            observation
            reward
            done
            info
        """

        if action_id < 0 or action_id >= num_actions():
            raise ValueError(f"Invalid action_id: {action_id}")

        before = get_player_position(self.pyboy)

        direction = get_action_name(action_id)
        moved = walk_tile(self.pyboy, direction, verbose=False)

        # Give the game a few frames to settle.
        run_frames(self.pyboy, 10)

        after = get_player_position(self.pyboy)

        reward = calculate_leave_house_reward(
            before=before,
            after=after,
            visited_positions=self.visited_positions,
        )

        self.visited_positions.add(position_key(after))

        self.step_count += 1

        reached_goal = after["map_id"] == PALLET_TOWN_MAP_ID
        ran_out_of_steps = self.step_count >= self.max_steps

        done = reached_goal or ran_out_of_steps

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
        """
        Return the current observation.

        For now, the observation is just map/x/y.

        Later we can add:
          screen pixels
          party HP
          battle state
          text box state
          inventory
        """

        return get_player_position(self.pyboy)

    def close(self):
        """
        Stop the emulator.
        """

        self.pyboy.stop()