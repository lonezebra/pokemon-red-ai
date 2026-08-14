from stable_baselines3 import DQN

from core.emulator import create_emulator, run_frames
from core.state import load_state
from core.config import PROJECT_ROOT
from core.pathfind import _step
from core.memory import get_player_position
from core.battle_runner import fight_current_battle
from core.controls import wait_for_free_movement
from actions import get_action_name, num_actions
from rewards.route3_rewards import (
    calculate_route3_reward,
    position_key,
    ROUTE_3_MAP_ID,
    MT_MOON_MAP_ID,
    KNOWN_TRAINER_TILES,
    _DISTANCES,
)

ROUTE_3_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "route3_leveled.state"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"


class PokemonRedRoute3Env:
    """
    Navigation across Route 3 toward its real exit: Mt. Moon (map 15).

    An earlier version of this env targeted a placeholder GOAL_TILE
    (22, 10), the deepest point a survey that turned out not to be
    exhaustive could reach, believing Mt. Moon was blocked by something
    this project had no way past yet. It wasn't -- see
    rewards/route3_rewards.py's module docstring for the real reason
    over 300 tiles (including the true exit) went undiscovered: Gen 1
    trainers only battle once, and the survey always re-tested each
    direction from a pre-battle snapshot, so it could never see what
    opens up once a blocking trainer is actually beaten. A human
    playthrough of the exact button sequence past the old "wall" is
    what found the real path.

    Structurally this is forest_env.py's pattern, not route1/route2's:
    Route 3 has forced trainer battles along the way (Gen 1 doesn't
    allow fleeing them), resolved transparently by the already-solved
    trainer-battle DQN, the same as the forest. KNOWN_TRAINER_TILES is
    derived differently though -- the forest's came from separately
    captured per-trainer states; Route 3's survey never captured those,
    so its tiles come straight from the survey's own graph (any tile
    whose recorded edge is not a plain 1-tile step -- see
    route3_rewards._load_trainer_trigger_tiles).

    Starts from route3_leveled.state (Lv20), not the pristine
    route3_entry.state (Lv13-14): the entry-level party measured a heal
    trip through this route's trainer gauntlet at up to 72% of max HP
    per fight, which is not survivable across the many repeated fights
    an RL policy's early, near-random exploration would trigger.

    No mid-episode healing, matching every other navigation env's
    fixed-budget convention -- HP only ever goes down over an episode. A
    blackout lands the player on Pewter's Pokemon Center map, which the
    reward function already penalizes as leaving Route 3, and `done`
    already covers landing anywhere else.
    """

    def __init__(self, max_steps=400):
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.step_count = 0
        self.visited_positions = set()
        self.probed_trainer_moves = set()
        self._blocked_streak = 0
        self._blocked_tile = None
        self.min_distance = None
        self.trainer_model = DQN.load(str(TRAINER_MODEL_PATH))

    def _handle_battle(self, pyboy):
        fight_current_battle(pyboy, self.trainer_model)
        wait_for_free_movement(pyboy)

    def _should_engage_trainer(self, before, direction):
        """Same contract and reasoning as forest_env.py's -- see there for
        the measured cost of an unconditional probe and the secondary-
        sighting-tile fallback this mirrors."""
        tile = (before["x"], before["y"])

        suspicious = self._blocked_streak >= 3
        if tile not in KNOWN_TRAINER_TILES and not suspicious:
            return False

        attempt = (tile, direction)
        if attempt in self.probed_trainer_moves:
            return False

        self.probed_trainer_moves.add(attempt)
        return True

    def reset(self):
        load_state(self.pyboy, ROUTE_3_ENTRY_STATE_PATH)
        run_frames(self.pyboy, 30)

        self.step_count = 0
        self.visited_positions = set()
        self.probed_trainer_moves = set()
        self._blocked_streak = 0
        self._blocked_tile = None
        self.min_distance = None

        position = get_player_position(self.pyboy)
        self._note_depth(position)
        self.visited_positions.add(position_key(position))

        return self._get_observation()

    def step(self, action_id):
        if action_id < 0 or action_id >= num_actions():
            raise ValueError(f"Invalid action_id: {action_id}")

        before = get_player_position(self.pyboy)
        direction = get_action_name(action_id)

        moved = _step(
            self.pyboy,
            direction,
            handle_battle=self._handle_battle,
            should_engage_trainer=self._should_engage_trainer,
        )

        after = get_player_position(self.pyboy)

        stuck_here = position_key(before) == position_key(after)
        if stuck_here and self._blocked_tile == position_key(before):
            self._blocked_streak += 1
        elif stuck_here:
            self._blocked_tile = position_key(before)
            self._blocked_streak = 1
        else:
            self._blocked_tile = None
            self._blocked_streak = 0

        reward = calculate_route3_reward(before=before, after=after)

        self.visited_positions.add(position_key(after))
        self._note_depth(after)
        self.step_count += 1

        reached_goal = after["map_id"] == MT_MOON_MAP_ID
        left_route_3 = after["map_id"] not in (ROUTE_3_MAP_ID, MT_MOON_MAP_ID)
        ran_out_of_steps = self.step_count >= self.max_steps

        done = reached_goal or left_route_3 or ran_out_of_steps

        info = {
            "moved": moved,
            "direction": direction,
            "before": before,
            "after": after,
            "reached_goal": reached_goal,
            "step_count": self.step_count,
            "min_distance": self.min_distance,
        }

        return self._get_observation(), reward, done, info

    def _note_depth(self, position):
        if position["map_id"] != ROUTE_3_MAP_ID:
            return
        distance = _DISTANCES.get((position["x"], position["y"]))
        if distance is None:
            return
        if self.min_distance is None or distance < self.min_distance:
            self.min_distance = distance

    def _get_observation(self):
        return get_player_position(self.pyboy)

    def close(self):
        self.pyboy.stop()
