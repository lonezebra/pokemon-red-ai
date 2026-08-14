# Segmented-skills approach (archived)

This is the project's original strategy, moved here as-is when the project
switched to an end-to-end whole-game PPO agent (see `../../README.md` and
`src/train_whole_game.py`). Nothing here was rewritten to make the move --
these are the exact same files, `git mv`'d, so `git log --follow` on any
file in here still shows its full history.

## What this was

One tabular-Q or DQN policy per task (leave the house, walk Route 1, win
the rival battle, ...), each trained and evaluated separately, then
chained into a single continuous run by `controller.py`. The main
`README.md`'s "What's actually implemented" section is the full writeup of
how this worked and what it achieved -- that history is left in place
there rather than duplicated here.

## Why it moved, not just this folder

The whole-game agent (`src/train_whole_game.py`) is now the primary
track. This code isn't deleted -- the results it produced (checkpoints,
Q-tables, verified save states) are real and still tracked in `models/`
and `saves/` at the project root, unchanged -- but it's not being
developed further, so it no longer belongs mixed in with `src/`.

## Known unresolved issue

`train_navigation_parallel.py`'s `POKEMON_RED_STOP_BY` deadline check has
a confirmed bug: a Route 3 training run with `POKEMON_RED_STOP_BY="19:00"`
did not stop at the deadline (rounds kept completing hours past it, and
the run was not manually restarted). This moved here unfixed. If this
track is ever revived, that's the first thing to chase down -- empirically
(a short throwaway run with a deadline a minute out), not by re-reading the
code again, since a careful code read during the same session it was found
did not turn up the cause.

## Running anything in here again

Every file kept its position *relative to the other files in this folder*
-- `envs/`, `rewards/`, and `agents/` are the same subpackages they were in
`src/`, so cross-references like `controller.py`'s
`from agents.skills import ...` need no changes and just work, the same as
they did before the move.

The one thing that doesn't resolve on its own is `core/` (shared
infrastructure -- emulator control, memory reading, save states -- used by
both this and the whole-game agent, so it stayed in `src/core/` rather
than moving here). Point `PYTHONPATH` at `src/` when running anything from
this folder:

```bash
cd archive/segmented-skills
PYTHONPATH=../../src ../../.venv/bin/python3 train_route3_agent.py
```

This was verified to actually work, not just assumed to: see the parent
session's verification pass, which ran `watch_route1_agent.py` this way
end to end before this README was written.

`render_forest_mashup.py` needs the same treatment for one more reason --
it imports `render_route1_mashup`, which also stayed in `src/` because the
whole-game track's `render_whole_game_runs.py` reuses it directly. Same
fix, same command shape.

`saves/*.state`, `models/*.json`, and `models/*_q_table.json` never moved
-- only code did. Every path these scripts already reference resolves
exactly as it did before, since `core/config.py`'s `PROJECT_ROOT` is
computed from `core/config.py`'s own location, not the caller's.

## `screenshots/`

Split the same way as the code, and for the same reason -- some of it is
still in active use by the whole-game track, most of it isn't:

- **Stayed in `src/`-adjacent `screenshots/`**: the map panoramas
  (`route1_map.png`, `forest_map.png`, `map2_map.png`,
  `map2_badged_map.png`, `map54_map.png`, `route3_map.png`, plus their
  `*_map_meta.json`) and the four `player_sprite_*.png` files. All of
  these are genuinely shared -- `src/render_whole_game_runs.py` reuses
  `src/render_route1_mashup.py` and `src/build_player_sprite.py` directly,
  and those load exactly these files. Moving them would have broken the
  whole-game track's own heatmaps and mashup GIFs.
- **`screenshots/` in this folder**: the ~180 git-tracked debug
  screenshots the old scouting/`create_*_state.py` scripts produced along
  the way (numbered probe frames, `oak_dialogue_after_a_*`,
  `choose_starter_*`, and similar), plus the three old mashup runs under
  `screenshots/mashups/`.
- **`screenshots/training_scratch/`**: ~580 per-round training-progress
  GIFs (`route3_progress_round001.gif` through `route3_progress_round464.gif`,
  `forest_progress_round*.gif`, `forest_curriculum_*.gif` -- about 96MB).
  These were never git-tracked -- the project's own `.gitignore` calls
  them out as disposable scratch that any later round supersedes -- so
  moving them preserves nothing git-history-wise, it's purely tidying
  `screenshots/` up. They're genuinely safe to delete outright if you'd
  rather reclaim the space; they were only moved instead of deleted because
  "archive" was the ask.
