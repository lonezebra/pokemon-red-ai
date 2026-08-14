import random
import re

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
    _DISTANCES,
)

FOREST_ENTRY_STATE_PATH = PROJECT_ROOT / "saves" / "leveled.state"
TRAINER_MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
TRAINER_BATTLE_STATE_DIR = PROJECT_ROOT / "saves" / "trainer_battles"


def _load_known_trainer_tiles():
    """
    The player-side tiles create_trainer_battle_states.py found blocked by
    a trainer, read back from the state files it captured there (one per
    distinct trainer, named trainer_<x>_<y>.state). Gating the expensive
    _try_engage_trainer probe (see core/pathfind.py) to only these tiles
    is safe specifically because that capture ran over the same
    exhaustively-surveyed forest (713/713 tiles, frontier fully
    exhausted) that this env trains on: any trainer, anywhere reachable,
    would already have been walked into and captured.
    """
    tiles = set()
    for path in TRAINER_BATTLE_STATE_DIR.glob("trainer_*.state"):
        match = re.match(r"trainer_(-?\d+)_(-?\d+)\.state", path.name)
        if match:
            tiles.add((int(match.group(1)), int(match.group(2))))
    return tiles


KNOWN_TRAINER_TILES = _load_known_trainer_tiles()


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

    def __init__(self, max_steps=1000, start_states=None):
        """
        start_states: optional list of save-state paths for reset() to
        choose from uniformly, instead of always starting at the forest
        entrance. This is what curriculum training varies -- see
        tools/build_curriculum_states.py for how the near-goal states are
        captured. None keeps the original entrance-only behaviour, so
        every existing caller is unaffected.
        """
        self.pyboy = create_emulator()
        self.max_steps = max_steps
        self.start_states = list(start_states) if start_states else [FOREST_ENTRY_STATE_PATH]
        self.start_state = None
        self.step_count = 0
        self.visited_positions = set()
        self.probed_trainer_moves = set()
        self._blocked_streak = 0
        self._blocked_tile = None
        self.trainer_model = DQN.load(str(TRAINER_MODEL_PATH))

    def _handle_battle(self, pyboy):
        fight_current_battle(pyboy, self.trainer_model)
        # Clears any trailing "you got $X" / trainer's parting line, win
        # or lose, the same as every other scripted interaction here.
        wait_for_free_movement(pyboy)

    def _should_engage_trainer(self, before, direction):
        """
        Pay for the trainer probe at most once per tile-and-direction per
        episode.

        Restricting it to known trainer tiles wasn't enough, because Gen 1
        leaves a defeated trainer standing on their tile, still blocking
        it. So after the fight is won the move stays blocked forever, and
        without this every later bump re-ran the full 12-press probe that
        now cannot possibly succeed.

        Measured rather than assumed, since the first estimate of this was
        badly wrong: a re-probed blocked bump costs 0.28s of wall time
        against 0.09s without, so roughly 3x, not the two orders of
        magnitude that "12 presses, ~11s of emulated time" suggests.
        Headless PyBoy runs about fifty times real-time, so emulated
        seconds are not wall seconds -- an easy conflation to make when
        reasoning about frame counts instead of measuring.
        tools/test_trainer_probe_cost.py is that measurement. 3x on a move
        a near-random policy retries constantly is worth having, but it is
        an efficiency fix and nothing more; it was not the cause of workers
        appearing stuck on a trainer.

        Keying on the direction as well as the tile fixes a second, quieter
        waste: a trainer-adjacent tile is usually a plain wall on its other
        sides, and the tile-only version paid the probe for those too.

        Once per episode is sufficient in both outcomes. If a live trainer
        is there, the probe finds them, the fight happens, and they are
        beaten -- probing again buys nothing. If it's a wall, it was a wall
        the first time. Recording the attempt here rather than in the
        caller keeps the bookkeeping next to the reason for it, at the cost
        of a predicate that isn't purely a question.
        """
        tile = (before["x"], before["y"])

        # A tile can be inside a trainer's line of sight without being in
        # KNOWN_TRAINER_TILES, because those were captured from a single
        # approach direction each: the trainer engaged from (2,19) also
        # sights a player arriving at (1,18), and walking onto such a
        # tile starts the sighting cutscene -- control locked until the
        # pre-battle dialogue is advanced, which only the probe's A
        # presses ever do. Without this clause an episode routed through
        # a secondary sighting tile was simply frozen until max_steps:
        # every direction blocked, the probe gated off, no Q-value able
        # to change what the env physically couldn't act on. Three
        # consecutive blocked steps on one tile is the signature (a
        # plain wall bump never repeats from the same tile that often
        # under epsilon-greedy without being frozen), so treat the tile
        # as suspect and pay for one probe.
        suspicious = self._blocked_streak >= 3
        if tile not in KNOWN_TRAINER_TILES and not suspicious:
            return False

        attempt = (tile, direction)
        if attempt in self.probed_trainer_moves:
            return False

        self.probed_trainer_moves.add(attempt)
        return True

    def reset(self):
        # Uniform over whatever the caller supplied; the single-entry
        # default makes this the original fixed reset.
        self.start_state = random.choice(self.start_states)
        load_state(self.pyboy, self.start_state)
        run_frames(self.pyboy, 30)

        self.step_count = 0
        self.visited_positions = set()
        # Cleared per episode: reset() reloads a state where every trainer
        # is undefeated again, so last episode's probes say nothing about
        # this one.
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

        # Feed the secondary-sighting detector in _should_engage_trainer:
        # count consecutive steps that failed to move off one tile. The
        # predicate runs mid-_step and sees the streak as of the previous
        # steps, which is the correct reading -- this step's own outcome
        # can't be known while it is still deciding whether to probe.
        stuck_here = position_key(before) == position_key(after)
        if stuck_here and self._blocked_tile == position_key(before):
            self._blocked_streak += 1
        elif stuck_here:
            self._blocked_tile = position_key(before)
            self._blocked_streak = 1
        else:
            self._blocked_tile = None
            self._blocked_streak = 0

        reward = calculate_forest_reward(before=before, after=after)

        self.visited_positions.add(position_key(after))
        self._note_depth(after)
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
            "min_distance": self.min_distance,
            "start_state": self.start_state.name,
        }

        return self._get_observation(), reward, done, info

    def _note_depth(self, position):
        """
        Track the closest-to-goal point this episode has reached, in
        shortest-path hops. tiles_visited turned out to be misleading as a
        progress readout: a policy following the corridor directly visits
        *fewer* tiles than one wandering shallowly, so a falling count can
        be improvement. Depth is monotone in what actually matters.
        """
        if position["map_id"] == CONNECTOR_MAP_ID:
            distance = 0
        elif position["map_id"] != FOREST_MAP_ID:
            return
        else:
            distance = _DISTANCES.get((position["x"], position["y"]))
            if distance is None:
                return
        if self.min_distance is None or distance < self.min_distance:
            self.min_distance = distance

    def _get_observation(self):
        return get_player_position(self.pyboy)

    def close(self):
        self.pyboy.stop()
