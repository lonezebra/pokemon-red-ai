import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.emulator import create_emulator, run_frames
from core.state import load_state, wild_encounter_state_path, WILD_ENCOUNTER_STATE_DIR
from core.controls import press_button, advance_battle_dialogue
from core.memory import (
    POKE_BALL_ITEM_ID,
    get_bag_item_quantity,
    get_detailed_battle_state,
    get_move_cursor_slot,
    get_party_count,
    randomize_battle_mon_stats,
    set_bag_item_quantity,
)
from rewards.wild_battle_rewards import calculate_wild_battle_reward


NUM_MOVE_SLOTS = 4
RUN_ACTION = 4
CATCH_ACTION = 5
NUM_ACTIONS = 6

# Plenty for a 30-step episode (no fight here has ever needed more than a
# couple of throws) without being large enough to matter for anything
# else. Written directly into the bag at reset -- every wild-encounter
# state predates this project needing an item, so all of them load with
# an empty bag, the same reason randomize_battle_mon_stats already
# rewrites battle stats directly rather than trusting the save file.
POKEBALLS_PER_EPISODE = 5


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

    Action 5 throws a Poke Ball. Every save state under
    saves/wild_encounters/ predates this project ever needing an item,
    so all of them load with an empty bag; reset() writes
    POKEBALLS_PER_EPISODE directly into it (set_bag_item_quantity),
    matching the existing convention of writing player-side facts
    straight into memory at reset rather than depending on the save
    file (randomize_battle_mon_stats already does this for battle
    stats). Throwing with zero balls left is treated as an invalid
    action, the same as naming an unusable move slot, and falls back to
    the first valid move instead -- not to running, so an empty-handed
    catch attempt still costs a real turn of fighting rather than a
    free pass. A catch is detected by the party's own Pokemon count
    increasing (get_party_count before vs after), the only signal that
    can't be confused with the battle simply ending some other way.

    The button sequence (down from FIGHT opens ITEM, per _attempt_run's
    documented menu layout; POKE BALL sits alone at the top of a bag
    reset() only ever puts one item in) was verified against real
    screenshots before being trusted, including both outcomes: "Aww!
    It appeared to be caught!" on a failed throw and a clean return to
    the overworld (party count 1 -> 2, one ball consumed) on a real
    catch.

    Catching does not award XP in Gen 1 (only a knockout does), so a
    flat catch bonus bigger than winning -- the first version of this
    env had exactly that -- would train the agent to try catching every
    single encounter forever, full party or not, duplicate or not,
    since nothing in the reward said otherwise. species_already_caught
    tracks which species this *training run* has already caught (a
    Python set on the env, not the game's own Pokedex flags: those are
    indexed by National Dex number rather than this codebase's internal
    species index throughout, and converting between them needs a full
    ~190-entry lookup table that would be pure transcription risk to
    hardcode from memory rather than something worth verifying here;
    the real Pokedex also has no meaningful "already owned" answer to
    give a fixed save-state reset every episode anyway, so an env-level
    set is both simpler and the more correct answer to what training
    actually needs). A new species catches at CATCH_REWARD; a species
    already in the set catches at the much smaller CATCH_DUPLICATE_
    REWARD, teaching the agent that catching is preferred only when it
    grows the collection, not reflexively.

    Persists across reset() by design, not cleared per episode --
    "already caught" is supposed to mean across the run, the same way
    a real Pokedex does, not reset to empty every 30 steps.

    Left for later, deliberately not addressed here: judging whether a
    specific individual's stats are "good enough" to bother catching.
    Gen 1 IVs aren't exposed as a clean fraction the way HP is, and
    "good enough compared to what" needs a real comparison basis (the
    rest of the party? every past catch of the same species?) that's
    its own design question, not a quick addition alongside this one.

    Observation: [your_hp_fraction, enemy_hp_fraction,
                  move1_valid, move2_valid, move3_valid, move4_valid,
                  poke_balls_remaining_fraction, species_already_caught]
    Deliberately not including species/level directly for this first
    version -- same reasoning as leave-house/rival-battle starting
    minimal: HP fractions and move availability might already be enough
    signal to learn sensible fight-or-flee behavior, and this is cheap
    to revisit if evaluation says otherwise. Ball count and already-
    caught *are* included, unlike moves/RUN which need no such signal:
    neither is inferable from the enemy/battle state already in the
    observation the way a move slot's own validity is.

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
        # Across the whole training run, not per-episode -- see the
        # class docstring for why this isn't cleared in reset().
        self.species_already_caught = set()

        self.encounter_paths = sorted(WILD_ENCOUNTER_STATE_DIR.glob("species_*.state"))
        if not self.encounter_paths:
            raise FileNotFoundError(
                f"No wild encounter states found in {WILD_ENCOUNTER_STATE_DIR}. "
                "Run create_wild_encounter_state.py first."
            )

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        encounter_path = random.choice(self.encounter_paths)
        load_state(self.pyboy, encounter_path)
        run_frames(self.pyboy, 10)

        if self.randomize_stats:
            randomize_battle_mon_stats(self.pyboy, random)

        set_bag_item_quantity(self.pyboy, POKE_BALL_ITEM_ID, POKEBALLS_PER_EPISODE)

        self.step_count = 0

        state = get_detailed_battle_state(self.pyboy)
        return self._observation(state), {}

    def step(self, action):
        before = get_detailed_battle_state(self.pyboy)
        party_count_before = get_party_count(self.pyboy)

        if action == RUN_ACTION:
            self._attempt_run()
            chose_invalid_action = False
        elif action == CATCH_ACTION:
            chose_invalid_action = get_bag_item_quantity(self.pyboy, POKE_BALL_ITEM_ID) <= 0
            if chose_invalid_action:
                # Falls back to fighting, not running -- naming an
                # unusable move slot gets the same treatment (the first
                # valid move, never a free pass), so an empty-handed
                # catch attempt costs a real turn the same way.
                self._select_move(self._valid_move_slots(before)[0])
            else:
                self._attempt_catch()
        else:
            valid_slots = self._valid_move_slots(before)
            chose_invalid_action = action not in valid_slots
            actual_action = valid_slots[0] if chose_invalid_action else action
            self._select_move(actual_action)

        after = get_detailed_battle_state(self.pyboy)
        caught = get_party_count(self.pyboy) > party_count_before
        # Checked before updating the set: a catch's reward depends on
        # whether the species was *already* known going into this catch,
        # not on the post-catch state where it always would be.
        caught_new_species = caught and before["enemy_mon_species"] not in self.species_already_caught
        reward = calculate_wild_battle_reward(
            before, after, invalid_action=chose_invalid_action,
            caught=caught, caught_new_species=caught_new_species,
        )
        if caught:
            self.species_already_caught.add(before["enemy_mon_species"])

        self.step_count += 1

        terminated = not after["in_battle"]
        truncated = self.step_count >= self.max_steps

        info = {
            "before": before,
            "after": after,
            "chose_invalid_action": chose_invalid_action,
            "caught": caught,
            "caught_new_species": caught_new_species,
            "won": terminated and after["enemy_mon_hp"] == 0 and not caught,
            "lost": terminated and after["battle_mon_hp"] == 0,
            "fled": terminated and after["enemy_mon_hp"] > 0
                    and after["battle_mon_hp"] > 0 and not caught,
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

    def _attempt_catch(self):
        # From FIGHT: down opens ITEM (see _attempt_run for the menu
        # layout this was verified against). reset() only ever puts one
        # item in the bag, so POKE BALL is always the top, default-
        # cursor entry -- confirmed live, including both outcomes: "Aww!
        # It appeared to be caught!" on a failed throw, and a clean
        # return to the overworld (party count +1, one ball consumed)
        # on a real catch. advance_battle_dialogue's own stopping
        # condition (battle menu open again, or battle over) already
        # covers whichever one happens.
        press_button(self.pyboy, "down", hold_frames=10, release_frames=15)
        press_button(self.pyboy, "a", hold_frames=10, release_frames=20)
        press_button(self.pyboy, "a", hold_frames=10, release_frames=20)
        run_frames(self.pyboy, 30)
        advance_battle_dialogue(self.pyboy)

    def _observation(self, state):
        your_fraction = state["battle_mon_hp"] / max(state["battle_mon_max_hp"], 1)
        enemy_fraction = state["enemy_mon_hp"] / max(state["enemy_mon_max_hp"], 1)

        move_valid = [
            1.0 if (state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0) else 0.0
            for i in range(NUM_MOVE_SLOTS)
        ]

        balls_fraction = (
            get_bag_item_quantity(self.pyboy, POKE_BALL_ITEM_ID) / POKEBALLS_PER_EPISODE
        )
        already_caught = 1.0 if state["enemy_mon_species"] in self.species_already_caught else 0.0

        return np.array(
            [your_fraction, enemy_fraction] + move_valid + [balls_fraction, already_caught],
            dtype=np.float32,
        )

    def close(self):
        self.pyboy.stop()
