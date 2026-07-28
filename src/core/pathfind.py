import io
from collections import deque

from core.emulator import run_frames
from core.controls import walk_tile, attempt_run_from_wild_battle, wait_for_position_to_settle
from core.memory import get_player_position, is_in_battle, get_battle_type

# General-purpose overworld pathfinding scaffolding.
#
# Every route in this project up to now was found by biased random walks
# with stuck-escape heuristics (see build_route1_map.py,
# create_route22_entry_state.py). That works, but it is slow, seed-
# dependent, and it failed outright on anything maze-like -- several
# thousand scripted steps never found Route 2's exit, and the same
# approach wandered into Viridian's buildings as often as its actual
# exits.
#
# This replaces that with a breadth-first search over real game states:
# from the current position, try all four directions, snapshot every
# newly reached tile, and keep expanding until a tile satisfies the
# caller's predicate. Two properties make it reliable:
#
#   - It is exhaustive within a map, so "no path exists" is a real
#     answer rather than "the random walk got unlucky". A BFS like this
#     is what established that Viridian City has no northern exit at
#     all (500 reachable tiles, exits only to Route 1, Route 22, and
#     four buildings).
#   - On success it *loads the snapshot it found* rather than replaying
#     the moves that got there. Replaying could diverge -- Route 1's
#     grass can interrupt any step with a wild encounter, and fleeing
#     consumes a different number of turns each time -- but restoring a
#     state that was already verified to be at the target cannot.
#
# Wild encounters during the search are fled automatically, the same way
# envs/route1_env.py does, so the search only ever deals with overworld
# positions.

DIRECTIONS = ("up", "down", "left", "right")

# Bounds worst-case memory: each visited tile holds a full save state
# (~170KB), so ~1200 tiles is roughly 200MB. Every overworld map this
# project has measured is well inside that (Viridian City is 500).
DEFAULT_MAX_TILES = 1200


def _snapshot(pyboy):
    buf = io.BytesIO()
    pyboy.save_state(buf)
    return buf.getvalue()


def _restore(pyboy, data):
    buf = io.BytesIO(data)
    buf.seek(0)
    pyboy.load_state(buf)
    run_frames(pyboy, 2)


def _step(pyboy, direction, handle_battle=None):
    """
    One tile move, fleeing any wild encounter it triggers.

    `handle_battle(pyboy)`, if given, is tried instead of fleeing when the
    encounter is a trainer battle (unlike a wild Pokemon, Gen 1 does not
    allow fleeing one at all -- without a handler, a trainer occupying a
    tile is simply unreachable, which is correct for any map whose
    trainers can't yet be beaten). If the handler clears the battle and
    the original move had been blocked, the step is retried once: a
    trainer's sightline trigger commonly blocks movement only until the
    trainer itself is dealt with, not because the tile is physically
    impassable.
    """
    before = get_player_position(pyboy)

    moved = walk_tile(pyboy, direction, verbose=False)
    run_frames(pyboy, 6)

    if is_in_battle(pyboy):
        if get_battle_type(pyboy) == 2 and handle_battle is not None:
            handle_battle(pyboy)
            if not moved:
                moved = walk_tile(pyboy, direction, verbose=False)
                run_frames(pyboy, 6)
        else:
            attempt_run_from_wild_battle(pyboy)

    # Crossing a map boundary (a door, or a route edge) hands control
    # back only after the game finishes auto-walking the player clear of
    # it -- the same settling problem controller.py hit at the house and
    # lab exits. Snapshotting mid-animation would store a position that
    # is about to change on its own.
    if get_player_position(pyboy)["map_id"] != before["map_id"]:
        wait_for_position_to_settle(pyboy)

    return moved


def walk_to(pyboy, predicate, max_tiles=DEFAULT_MAX_TILES, stay_on_map=True, handle_battle=None):
    """
    Search outward from the player's current position until reaching a
    tile where `predicate(position_dict)` is true, then leave the
    emulator standing on that tile.

    `stay_on_map=True` (the default) treats a map change as a wall: the
    search notes it and backs off rather than expanding through it. That
    keeps a search bounded to the current map, so looking for "the tile
    that exits to Route 1" doesn't wander off into Route 1 and start
    exploring that too. Pass False to search across map boundaries.

    `handle_battle`, if given, is passed through to `_step` -- see there
    for what it's for.

    Returns True and leaves the player at the target, or returns False
    and leaves the player where they started.
    """

    origin = _snapshot(pyboy)
    start = get_player_position(pyboy)

    if predicate(start):
        return True

    start_map = start["map_id"]
    start_key = (start["map_id"], start["x"], start["y"])

    states = {start_key: origin}
    seen = {start_key}
    queue = deque([start_key])

    while queue and len(seen) < max_tiles:
        key = queue.popleft()

        for direction in DIRECTIONS:
            _restore(pyboy, states[key])
            moved = _step(pyboy, direction, handle_battle=handle_battle)
            position = get_player_position(pyboy)

            changed_map = position["map_id"] != start_map
            if not moved and not changed_map:
                continue
            if changed_map and stay_on_map:
                if predicate(position):
                    return True
                continue

            next_key = (position["map_id"], position["x"], position["y"])
            if next_key in seen:
                continue

            snapshot = _snapshot(pyboy)
            if predicate(position):
                return True

            seen.add(next_key)
            states[next_key] = snapshot
            queue.append(next_key)

    _restore(pyboy, origin)
    return False


def survey_map(pyboy, max_tiles=DEFAULT_MAX_TILES, on_visit=None, handle_battle=None):
    """
    Exhaustively flood-fill the map the player is standing on, without
    stepping off it, and report what is actually there:

        tiles -- every reachable (x, y) on this map
        exits -- {((x, y), direction): (map_id, x, y)} for each tile that
                 leads somewhere else

    `on_visit(pyboy, x, y)`, if given, is called once per newly reached
    tile while the emulator is actually standing on it -- which is what
    lets the map-panorama builder grab a screenshot of every tile rather
    than only the ones a random walk happened to cross.

    `handle_battle(pyboy)`, if given, is passed through to `_step` so a
    trainer occupying a tile can be fought and beaten rather than simply
    read as a wall -- see `_step` for why that only applies to trainer
    battles, never wild ones.

    If the search finishes before hitting `max_tiles`, the result is
    complete: anything absent from `exits` genuinely is not reachable
    from here. That is the whole point -- it is what turned "we never
    found Viridian's north exit" into "Viridian has no north exit", and
    the same distinction would have caught the Route 22 mix-up
    immediately (a route with no forward exit is obvious in one survey,
    but invisible across 1500 training episodes).

    Leaves the player where they started.
    """

    origin = _snapshot(pyboy)
    start = get_player_position(pyboy)
    start_map = start["map_id"]
    start_key = (start["x"], start["y"])

    states = {start_key: origin}
    tiles = {start_key}
    exits = {}
    queue = deque([start_key])

    if on_visit is not None:
        on_visit(pyboy, start["x"], start["y"])

    while queue and len(tiles) < max_tiles:
        key = queue.popleft()

        for direction in DIRECTIONS:
            _restore(pyboy, states[key])
            moved = _step(pyboy, direction, handle_battle=handle_battle)
            position = get_player_position(pyboy)

            if position["map_id"] != start_map:
                exits[(key, direction)] = (
                    position["map_id"],
                    position["x"],
                    position["y"],
                )
                continue
            if not moved:
                continue

            next_key = (position["x"], position["y"])
            if next_key not in tiles:
                tiles.add(next_key)
                states[next_key] = _snapshot(pyboy)
                queue.append(next_key)
                if on_visit is not None:
                    on_visit(pyboy, position["x"], position["y"])

    complete = not queue
    _restore(pyboy, origin)
    return tiles, exits, complete


def walk_to_tile(pyboy, x, y, **kwargs):
    return walk_to(pyboy, lambda p: p["x"] == x and p["y"] == y, **kwargs)


def walk_to_map(pyboy, map_id, **kwargs):
    """
    Cross into `map_id`, e.g. stepping through a building door or a route
    boundary.

    Keeps the default stay_on_map=True on purpose: the search should
    hunt for the doorway across the *current* map and step through it,
    not treat every other neighbouring map as more territory to explore.
    That means the target map has to border the current one -- callers
    chain one leg at a time rather than asking for a destination several
    maps away.
    """
    return walk_to(pyboy, lambda p: p["map_id"] == map_id, **kwargs)
