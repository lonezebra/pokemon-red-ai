from stable_baselines3 import DQN

from core.config import PROJECT_ROOT
from core.battle_runner import fight_current_battle
from core.controls import wait_for_free_movement
from core.memory import get_player_position
from build_map_panorama import build

# Re-runs the Viridian Forest survey now that its trainers are beatable.
#
# The first survey (see README, "Viridian Forest is surveyed but
# blocked") found only 676 reachable tiles and an exit list that led
# straight back to Route 2 -- because a trainer occupies its tile,
# survey_map()'s plain flood fill correctly reads one as a wall, the same
# as any other unwalkable tile. That result was real, but it was a
# statement about what's reachable *without* fighting, not a statement
# about the forest's actual layout. Now that a Lv10 party with the
# trainer-battle DQN evaluates at 100/100, the survey can be handed a
# battle handler that fights and wins instead of giving up -- so this
# finds the forest's true exit toward Pewter City, if it has one.

MODEL_PATH = PROJECT_ROOT / "models" / "trainer_battle_dqn.zip"
FOREST_MAP_ID = 51


def make_handle_battle(model):
    def handle_battle(pyboy):
        fight_current_battle(pyboy, model)
        # Clears any trailing "you got $X" / trainer's parting line --
        # the same trailing dialogue create_leveled_state.py's grind()
        # clears after every battle, win or lose.
        wait_for_free_movement(pyboy)

        position = get_player_position(pyboy)
        if position["map_id"] != FOREST_MAP_ID:
            # A loss blacks the party out to the last Pokemon Center,
            # which would otherwise look to the BFS like a legitimate
            # exit from this tile -- so the whole run is aborted rather
            # than silently mislabeling a loss as forest topology. At
            # 100/100 in evaluation this should not happen, but the
            # policy has only ever been measured against the five
            # trainers captured before Route 2 forced their discovery
            # to stop, and there may be others further in.
            raise RuntimeError(
                f"Lost a trainer battle during the survey -- blacked out "
                f"to map {position['map_id']}. The trained policy is not "
                f"reliable enough to survey past this trainer."
            )

    return handle_battle


def main():
    model = DQN.load(str(MODEL_PATH))
    build("leveled", "forest", handle_battle=make_handle_battle(model))


if __name__ == "__main__":
    main()
