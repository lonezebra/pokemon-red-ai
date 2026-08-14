from core.emulator import create_emulator, run_frames
from core.state import load_state, save_state, STARTER_OBTAINED_STATE_PATH, ROUTE_1_ENTRY_STATE_PATH
from core.config import PROJECT_ROOT
from core.controls import walk_tile, wait_for_position_to_settle
from core.memory import get_player_position, get_battle_state, print_player_position
from core.screen import save_screenshot
from create_rival_battle_state import walk_to_rival_trigger_and_battle
from create_starter_obtained_state import wait_for_control_and_walk
from agents.skills import RivalBattleSkill


# Findings from probing this by hand, starting at starter_obtained.state
# and playing through the already-solved rival battle:
#
#   - Winning the rival battle does not return you to the overworld
#     immediately -- there's a post-battle dialogue sequence to clear
#     first (Blue's "reaction" text), the same class of thing as the
#     pre-battle dialogue elsewhere in this project. wait_for_control_and_walk
#     (press A, test real movement, stop the instant it works) handles
#     it the same way it handles every other dialogue-then-movement
#     transition in this codebase.
#   - Oak's Lab's exit door does NOT put you back at outside_house.state's
#     (5, 6) -- that's the *player's house* door. Oak's Lab is a
#     different building elsewhere in Pallet Town, so walking out of it
#     lands you at a different spot: (map 0, x=12, y=12), one tile below
#     the lab's exterior door. wait_for_position_to_settle is still
#     needed here for the same reason as the player's-house exit: the
#     game keeps auto-walking a couple more tiles on its own afterward.
#   - Route 1's entrance is NOT a straight shot north from the lab door
#     (that column is blocked by a hedge one tile up). The actual gap in
#     the hedge is further west, at the same column (x=10) as Oak's own
#     "Hey! Wait!" trigger tile -- but you have to go *around* the lab
#     building to reach it, since the direct route (2 tiles left, then
#     straight up) hits the hedge almost immediately. Verified tile by
#     tile: right x4 clears the lab building's east side, up x10 is a
#     fully open column up to Pallet Town's top row, left x6 slides back
#     over to the x=10 gap, and up x3 from there crosses out of Pallet
#     Town (map 0) into Route 1 (map 12).
#   - Confirmed empirically that map 12 is Route 1 by cross-referencing
#     the public pret/pokered map-constants list (ROUTE_1 = $0C = 12),
#     and by the tall-grass-lined screenshot matching Route 1's known
#     look.

ROUTE_TO_LAB_EXIT = ["down"]
ROUTE_LAB_EXIT_TO_ROUTE_1 = ["right"] * 4 + ["up"] * 10 + ["left"] * 6 + ["up"] * 3

ROUTE_1_MAP_ID = 12


def walk_out_of_lab_and_up_to_route_1(pyboy):
    """
    From a live session right after winning the rival battle inside
    Oak's Lab (e.g. continuing straight from
    create_rival_battle_state.walk_to_rival_trigger_and_battle() plus
    playing the battle out, rather than reloading rival_battle.state),
    clear the post-battle dialogue, walk out of the lab, and continue on
    to Route 1's entrance.
    """

    if not wait_for_control_and_walk(pyboy, ROUTE_TO_LAB_EXIT[0]):
        print("Warning: never regained control after the rival battle.")
        return False

    for _ in range(15):
        moved = walk_tile(pyboy, "down", verbose=False)
        run_frames(pyboy, 10)
        if get_player_position(pyboy)["map_id"] != 40:
            break
    else:
        print("Warning: never left Oak's Lab through the south exit.")
        return False

    wait_for_position_to_settle(pyboy)

    for direction in ROUTE_LAB_EXIT_TO_ROUTE_1:
        walk_tile(pyboy, direction, verbose=False)
        run_frames(pyboy, 10)

    pos = get_player_position(pyboy)
    if pos["map_id"] != ROUTE_1_MAP_ID:
        print(f"Warning: expected to reach map {ROUTE_1_MAP_ID} (Route 1), got {pos}.")
        return False

    return True


def main():
    # Deferred import: controller.py imports walk_out_of_lab_and_up_to_route_1
    # from this module, so importing controller.py at module load time
    # here would be circular. main() is only run from the command line
    # (never imported), so importing it lazily here breaks the cycle
    # without needing to duplicate _battle_observation/_select_battle_move.
    from controller import _battle_observation, _select_battle_move, NUM_MOVE_SLOTS

    pyboy = create_emulator()

    print("Loading starter_obtained state...")
    load_state(pyboy, STARTER_OBTAINED_STATE_PATH)
    run_frames(pyboy, 60)

    print("Walking to the rival's trigger and into battle...")
    if not walk_to_rival_trigger_and_battle(pyboy):
        print("Warning: did not reach the battle as expected.")
        pyboy.stop()
        return

    print("Playing the rival battle with the trained DQN...")
    skill = RivalBattleSkill(PROJECT_ROOT / "models" / "rival_battle_dqn.zip")

    for _ in range(30):
        state = get_battle_state(pyboy)
        if not state["in_battle"]:
            break

        valid_slots = [
            i
            for i in range(NUM_MOVE_SLOTS)
            if state["battle_mon_moves"][i] != 0 and state["battle_mon_pp"][i] > 0
        ]
        action = skill.choose_action(_battle_observation(state))
        actual_action = action if action in valid_slots else valid_slots[0]
        _select_battle_move(pyboy, actual_action)

    final_state = get_battle_state(pyboy)
    if final_state["enemy_mon_hp"] != 0:
        print("Warning: did not win the rival battle.")
        pyboy.stop()
        return

    print("Won the rival battle. Walking out of the lab and up to Route 1...")
    if not walk_out_of_lab_and_up_to_route_1(pyboy):
        pyboy.stop()
        return

    print_player_position(pyboy, "Position at Route 1's entrance")
    save_screenshot(pyboy, "route_1_entry.png")

    print()
    print(f"Saving state to {ROUTE_1_ENTRY_STATE_PATH}...")
    save_state(pyboy, ROUTE_1_ENTRY_STATE_PATH)

    print("Done. Check screenshots/route_1_entry.png to confirm it shows")
    print("the player standing in Route 1's tall grass just past Pallet Town.")

    pyboy.stop()


if __name__ == "__main__":
    main()
