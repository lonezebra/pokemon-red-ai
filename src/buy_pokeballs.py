import os

# Headless before any core import -- see train_forest_agent.py.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

import sys  # noqa: E402

from core.config import PROJECT_ROOT  # noqa: E402
from core.controls import press_button  # noqa: E402
from core.emulator import create_emulator, run_frames  # noqa: E402
from core.memory import POKE_BALL_ITEM_ID, get_bag_item_quantity, get_money  # noqa: E402
from core.pathfind import walk_to_map, walk_to_tile  # noqa: E402
from core.state import load_state, save_state  # noqa: E402

# Buys Poke Balls from Pewter's Mart -- the party has been able to fight
# since the beginning of this project, but had never actually been able
# to *catch* anything: every start state up to this point carries zero
# Poke Balls, because nothing before this needed one.
#
# The clerk's position and the whole button sequence below were found
# by brute-force sweep, not guessed: every reachable tile in the Mart's
# small room was tried facing all four directions with a screenshot
# pixel-diff after each 'A' press, since a wandering customer NPC kept
# intercepting hand-picked approach angles with unrelated dialogue
# ("A shady old man got me to buy...") that looks identical in a plain
# text-box check to a real shop menu. The clerk turned out to be at
# (1, 5), reached by standing at (2, 5) facing left.
#
# Button sequence to complete a purchase, each step confirmed against
# an actual screenshot before being relied on:
#   A (greeting) -> A (BUY/SELL/QUIT, selects BUY on the default
#   cursor) -> A (item list, selects POKE BALL on the default cursor)
#   -> up x(n-1) (quantity, starts at x01) -> A (price prompt) ->
#   A (advances to the YES/NO confirmation) -> A (confirms YES on the
#   default cursor).
MART_STATE_PATH = PROJECT_ROOT / "saves" / "pewter_mart_entry.state"
PEWTER_CITY_MAP_ID = 2
MART_MAP_ID = 56
CLERK_APPROACH_TILE = (2, 5)

POKE_BALL_PRICE = 200


def buy_pokeballs(pyboy, count):
    """
    Assumes the player is already standing at CLERK_APPROACH_TILE,
    facing left, with the shop menu not yet open. Returns True on a
    confirmed purchase (bag count increased by exactly `count`).
    """
    before = get_bag_item_quantity(pyboy, POKE_BALL_ITEM_ID)

    press_button(pyboy, "left", hold_frames=12, release_frames=15)
    run_frames(pyboy, 10)
    for _ in range(3):
        # greeting -> BUY/SELL/QUIT (selects BUY) -> item list (selects
        # POKE BALL, the top and default-cursor entry both times)
        press_button(pyboy, "a", hold_frames=12, release_frames=20)
        run_frames(pyboy, 30)

    press_button(pyboy, "a", hold_frames=12, release_frames=20)
    run_frames(pyboy, 30)

    for _ in range(count - 1):
        press_button(pyboy, "up", hold_frames=10, release_frames=15)
    run_frames(pyboy, 10)

    press_button(pyboy, "a", hold_frames=12, release_frames=20)  # -> price prompt
    run_frames(pyboy, 30)
    press_button(pyboy, "a", hold_frames=12, release_frames=20)  # -> YES/NO
    run_frames(pyboy, 30)
    press_button(pyboy, "a", hold_frames=12, release_frames=20)  # confirm YES
    run_frames(pyboy, 40)

    # Clear the "Here you are!"/"Anything else?" trailing dialogue and
    # back out of the shop menu to QUIT, so the caller gets control back
    # in the overworld rather than mid-menu.
    for _ in range(6):
        press_button(pyboy, "b", hold_frames=12, release_frames=20)
        run_frames(pyboy, 20)

    after = get_bag_item_quantity(pyboy, POKE_BALL_ITEM_ID)
    return after == before + count


def main():
    target_state = sys.argv[1] if len(sys.argv) > 1 else "route3_leveled"
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    pyboy = create_emulator()
    load_state(pyboy, PROJECT_ROOT / "saves" / f"{target_state}.state")
    run_frames(pyboy, 30)

    money = get_money(pyboy)
    cost = count * POKE_BALL_PRICE
    print(f"Money: {money}, buying {count} Poke Balls for {cost}")
    if cost > money:
        print(f"Not enough money for {count} -- need {cost}, have {money}")
        pyboy.stop()
        return 1

    walk_to_map(pyboy, PEWTER_CITY_MAP_ID, max_tiles=2000)
    walk_to_map(pyboy, MART_MAP_ID, max_tiles=1000)
    walk_to_tile(pyboy, *CLERK_APPROACH_TILE, stay_on_map=True, max_tiles=200)

    ok = buy_pokeballs(pyboy, count)
    print(f"Purchase {'succeeded' if ok else 'FAILED'}: "
          f"now holding {get_bag_item_quantity(pyboy, POKE_BALL_ITEM_ID)} Poke Balls, "
          f"{get_money(pyboy)} money left")

    walk_to_map(pyboy, PEWTER_CITY_MAP_ID, max_tiles=1000)
    if target_state == "route3_leveled":
        from survey_route3 import ROUTE_3_MAP_ID
        walk_to_map(pyboy, ROUTE_3_MAP_ID, max_tiles=2000)

    output_path = PROJECT_ROOT / "saves" / f"{target_state}_with_balls.state"
    save_state(pyboy, output_path)
    print(f"Saved {output_path}")
    pyboy.stop()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
