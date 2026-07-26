# Pokemon Red AI

**The goal:** teach a computer to play Pokemon Red on its own — starting with
walking out of your bedroom, and eventually beating Brock, the first Gym
Leader. Long term, if it keeps working, push it toward the Elite Four.

This file explains what's actually built right now, why it's built that way,
and where it's headed. It's written so that even if you've never trained an
AI model or written a line of Python, you can follow along and understand
what's happening — and hopefully get curious enough to run it yourself.

## The big idea

Most "AI plays Pokemon" projects you might picture either hard-code every
move, or let a giant neural network stare at raw pixels and figure everything
out from scratch. This project deliberately avoids both extremes.

Instead, the rule here is:

> **Scripts are only allowed as scaffolding. The actual playing has to be
> learned.**

Concretely, that means:

- It's fine to write a small script that manually walks the character
  somewhere once, so we can discover *what the game's memory looks like*
  when the character is standing in a doorway, or *what coordinates* the
  house exit is at.
- It's fine to save the game's state right after a milestone (like "just
  woke up in the bedroom") so we don't have to sit through the same intro
  every time we test something.
- It is **not** the goal to have a permanent hard-coded script walk the
  character through the entire game. Once we know the coordinates and
  rules, the plan is to hand the task to a learning agent — something that
  tries actions, gets rewarded or penalized, and gradually gets better —
  rather than following a fixed script forever.

So the project grows in small loops:

```
small scripted probe → known fact about the game → small trainable task → learned behavior → next milestone
```

## How does an AI "see" a Game Boy game?

Two options exist: watch the raw pixels on screen, or read the game's
working memory directly (the Game Boy equivalent of RAM). This project
reads memory.

Why? Because Pokemon Red keeps simple facts about the world in fixed,
known memory locations — which map you're on, and your X/Y position on
that map, for instance. Reading three specific memory addresses tells you
exactly where the character is standing, instantly and perfectly, with no
guesswork. Trying to figure that out from a picture of the screen would be
far harder and far less reliable, especially this early in the project.

The emulator used is [PyBoy](https://github.com/Baekalfen/PyBoy), a Game
Boy emulator written in Python that lets a program read memory and press
buttons the same way a person would with a controller.

## What's actually implemented right now

Everything below exists in this repository today and runs. The `src/`
folder is split into two kinds of files: reusable library code (grouped
into `core/`, `envs/`, `rewards/` subfolders) and small scripts you run
directly (kept flat in `src/` so every command below stays simple —
`python src/whatever.py`, no extra package-path setup needed).

### Talking to the emulator (`src/core/`)

- **`src/core/emulator.py`** starts PyBoy, points it at your Pokemon Red
  ROM (which you provide yourself — see below), and sets how fast it runs.
  `run_frames()` just advances the emulator a given number of frames —
  Pokemon Red runs at roughly 60 frames per second, so `run_frames(pyboy, 60)`
  is "wait about one second."
- **`src/core/controls.py`** presses and releases Game Boy buttons. Two
  things in here matter beyond simple button presses:
  - `walk_tile()`: instead of holding a direction for a fixed amount of
    time (which sounds simple but isn't — too short and the character
    just turns to face that way without moving, too long and it walks
    multiple tiles), it holds the button and checks the character's
    position *every single frame* until the position actually changes,
    then releases. This was a real bug fixed early on, and it's the
    reason movement in this project is precise instead of flaky.
  - `advance_battle_dialogue()`: the battle equivalent of the same idea.
    Different moves produce different amounts of on-screen text (a
    stat-lowering move adds an extra message a plain damage move
    doesn't), so a fixed number of "press A to clear the text" presses
    turned out to be unreliable — it was actually caught silently
    executing the wrong move partway through the battle work below. This
    function presses A and checks the real game state after every press
    instead of guessing a count.
- **`src/core/memory.py`** reads facts directly out of the Game Boy's
  memory rather than trying to interpret the screen picture: which map
  you're on and your X/Y position, and (added for the battle milestone)
  whether a battle is active, both Pokemon's current/max HP, and each
  known move's ID and remaining PP. It also knows how to find the battle
  menu's cursor position and confirm the FIGHT/ITEM/RUN menu is currently
  on screen — both needed to reliably control a battle turn by turn (see
  below for why that turned out to be trickier than it sounds).
- **`src/core/state.py`** saves and loads PyBoy save states — snapshots of
  the entire game at one instant. Several exist now, each captured at a
  meaningful checkpoint: standing in the bedroom, standing outside the
  house, arriving at Oak's Lab, having just picked a starter Pokemon, and
  the moment the first rival battle begins. Resetting to any of these
  takes about a second, instead of replaying everything before it.
- **`src/core/screen.py`** saves a screenshot of the current game screen.
  Used throughout for visually double-checking that a script (or a memory
  reading) actually reflects what's really happening on screen.

### Turning "leave the house" into something an AI can practice (`src/actions.py`, `src/rewards/leave_house_rewards.py`, `src/envs/simple_env.py`)

The first task built this way, and still the simplest:

- **`src/actions.py`** defines what the agent is allowed to do in this
  task: move up, down, left, or right. Nothing else — no menus, no
  buttons like A or B. Keeping the choices small keeps the learning
  problem small.
- **`src/rewards/leave_house_rewards.py`** is the scoring rule. Every step
  costs a tiny penalty (encourages shorter paths), standing still costs
  more (discourages walking into walls), visiting a new tile for the
  first time earns a small reward (encourages exploring instead of pacing
  back and forth), reaching the downstairs area earns more, and reaching
  Pallet Town outside — the actual goal — earns a large reward.
- **`src/envs/simple_env.py`** ties it all together into a loop that
  should look familiar if you've heard of reinforcement learning before:
  `reset()` loads the bedroom save state and hands back the starting
  position; `step(action)` performs one move, checks the new position,
  computes the reward, and reports back whether the episode is finished
  (either the goal was reached, or too many steps passed). Nothing in
  here is Pokemon-specific logic telling the character what to do — it's
  just the scoreboard and rulebook. Whatever plays through this loop is
  the thing that has to actually figure out the path.
- **`src/run_random_agent.py`** plays that environment by picking
  completely random moves. It isn't meant to be smart — it exists to
  prove the environment itself works correctly (rewards make sense, the
  episode ends when it should, nothing crashes) *before* plugging in
  something that actually learns.
- **`src/agents/q_learning_agent.py`** is the actual learner: a tabular
  Q-learning agent (state = `(map_id, x, y)`, one value per action per
  state, updated from trial and error). **`src/train_q_agent.py`** trains
  it and **`src/watch_q_agent.py`** evaluates it. Current result: **30/30
  wins**, taking the same 19-step path every time.

  Getting there caught a real bug and a real under-training problem,
  both only visible by actually evaluating the trained policy instead of
  trusting the training-time success counter:
  - `leave_house_rewards.py`'s downstairs bonus (+5) originally checked
    the *current* map every step rather than whether the agent had just
    *arrived* there. Since reaching downstairs doesn't end the episode,
    the agent could sit there collecting +5 per step instead of pushing
    on — a training run "succeeding" 62/500 times was actually mostly
    measuring how well it learned to loiter, not to finish, given away
    by episode rewards (500-970) far higher than the reward design
    should ever produce.
  - Once that was fixed, a 500-episode run *looked* fine during training
    but its resulting policy failed every single evaluation episode,
    stuck repeating a wall-bump forever at one downstairs tile. Its
    learned values for that state were nearly identical across all four
    possible actions — undecided noise, not a real preference. Q-learning
    only updates a state when the agent actually passes through it, and
    reaching downstairs was itself infrequent early on, so everything
    past it got far fewer updates than the bedroom did. More episodes
    (2000) and slower exploration decay, giving those deeper states more
    chances to actually be learned, fixed it.

### Reaching, and beating, the first rival battle (`src/rewards/battle_rewards.py`, `src/envs/battle_env.py`, `src/train_battle_agent.py`, `src/watch_battle_agent.py`)

This milestone is further along than the one above — it's fully built,
trained, and verified:

- **`src/create_rival_battle_state.py`** walks from `starter_obtained.state`
  toward the lab exit and discovered something the original plan got
  wrong: the rival doesn't wait outside the lab, he stops the player
  *inside* Oak's Lab itself, right before the exit. The script replays
  the verified route and dialogue timing and saves `rival_battle.state`
  at the very first FIGHT/ITEM/RUN menu of the battle — before either
  side has moved — which is the natural "start of episode" point for
  training.
- **`src/envs/battle_env.py`** is a proper
  [Gymnasium](https://gymnasium.farama.org/) environment (the standard
  interface most reinforcement-learning tools expect), reset from
  `rival_battle.state`. Its action is simply "use move slot 0-3." Picking
  a move slot your Pokemon doesn't know yet, or has run out of PP for,
  costs a small penalty instead of pressing any button — a form of
  keeping the choices the agent can make matched to what's actually
  legal, the same spirit as the movement task above.
- **`src/rewards/battle_rewards.py`** scores each turn by the fraction of
  each side's HP that changed (so it means the same thing regardless of
  the Pokemon's actual HP total), plus a small per-turn penalty, plus one
  large win/loss bonus at the end that's deliberately bigger than
  anything the per-turn scoring could add up to — so the agent is never
  tempted to prolong a winnable fight instead of just finishing it.
- **`src/train_battle_agent.py`** trains a small neural network (a DQN,
  via the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
  library) to play this environment. Unlike the movement task, this uses
  a neural network instead of a lookup table, because battle situations
  (HP totals, which moves are available, outcomes) don't compress into a
  small table the same way three coordinates do.
- **`src/watch_battle_agent.py`** loads a trained model and plays 100
  battles with learning turned off, to measure how good it actually is.
  **Current result: 100/100 wins** (the project's bar for "good enough to
  move on" was 90/100).

Two real bugs were caught and fixed while building this, both by testing
against known ground truth instead of trusting the first version that
ran without crashing:

- The battle menu's move cursor turned out to be **sticky** (it
  remembers the last move you used, rather than resetting each turn) and
  the move list **wraps around** instead of stopping at the top/bottom.
  A script that just pressed "up" a few times to "reset" the cursor was
  actually landing on the wrong move most of the time — caught by
  checking which move's PP actually went down, not just trusting the
  intended button sequence.
- See `advance_battle_dialogue()` above for the second one.

### Scouting scripts (the scaffolding, not the destination)

A handful of scripts were how the coordinates and routes above were
actually discovered, and are kept around as reference/history rather than
things meant to run forever:

- **`src/leave_bedroom_bot.py`**, **`src/leave_bedroom_with_navigation.py`**,
  **`src/leave_house.py`** — earlier, increasingly cleaned-up versions of
  "manually walk a fixed route out of the house," used to nail down the
  exact route and confirm it works before it became the target the RL
  environment above tries to reach on its own.
- **`src/navigation.py`** — a simple "walk toward this X/Y coordinate"
  helper that fixes X first, then Y. It doesn't understand walls or
  obstacles yet, which is fine for its current use as a probing tool, but
  is exactly the kind of hard-coded logic this project intends to replace
  with learned behavior over time.
- **`old_tests/`** — earlier, rougher one-off debugging scripts (memory
  probes, movement calibration, brute-force route searches). Kept for
  history, not actively used.

## What's designed but not built yet (the roadmap)

Here's where this is headed, and why each step is designed the way it is.

1. ~~Actually train a Q-learning agent for the leave-house task.~~ **Done**
   — see `agents/q_learning_agent.py` above. 30/30 in evaluation, a
   19-step path every time.
2. ~~Chain more of the game's opening: reaching Professor Oak, choosing a
   starter Pokemon, and the forced first battle against your rival.~~
   **Done** — `saves/starter_obtained.state` and
   `saves/rival_battle.state` exist, and `create_rival_battle_state.py`
   documents exactly how the rival-battle trigger works.
3. ~~Battles need a different kind of learner than walking does.~~ **Done**
   — see `battle_env.py`/`train_battle_agent.py` above. The rival battle
   is currently beaten reliably (100/100 in evaluation), by a small
   neural network, not a lookup table, matching the reasoning that HP
   totals and move outcomes don't compress into a small table the way
   three coordinates do.
4. **A hand-written "controller"** will chain the individually trained
   skills together for a real end-to-end run — run the walking skill
   until a battle starts, then hand control to the battle skill, then
   hand control back — the same philosophy as everything above, just one
   level up: script the handoffs, learn the behavior. There are now two
   genuinely trained skills (leave-house, rival-battle) to actually chain
   together, so this is the natural next piece.
5. **Wild Pokemon encounters** are a different flavor of battle from the
   rival fight — the opponent varies, and unlike a rival fight you
   actually *can* run away — so they'll get their own environment variant
   rather than being forced into the current one. Until that exists, the
   plan is for the future controller (item 4) to fall back to a simple
   scripted "always use move 1" behavior for any battle type it doesn't
   have a trained policy for yet (e.g. a wild encounter met while
   navigating Route 1), just to survive it and resume navigation.
6. **Later still**: healing strategy (when to retreat/heal rather than
   push through a fight), and eventually eight badges and the Elite Four
   — each one added only once the step before it is actually working, not
   designed for prematurely.

## Try it yourself

You'll need your own legally-owned Pokemon Red ROM — this project
deliberately never includes or distributes one.

```bash
# 1. Set up your Python environment (once)
python3 -m venv .venv
source .venv/bin/activate
pip install pyboy numpy pillow gymnasium stable-baselines3

# 2. Add your ROM
#    Place it at: roms/pokemon_red.gb

# 3. Watch the leave-house environment work with a random (not-yet-smart) agent
python src/run_random_agent.py

# 4. Recreate the rival-battle save state (needs saves/starter_obtained.state)
python src/create_rival_battle_state.py

# 5. Train the battle DQN (headless/fast; drop the env var for a visible window)
POKEMON_AI_WINDOW_MODE=null python src/train_battle_agent.py

# 6. Evaluate it over 100 battles
POKEMON_AI_WINDOW_MODE=null python src/watch_battle_agent.py
```

The first time you run something that needs `saves/bedroom.state`, you'll
need to create it — that's a save state captured right after starting a
new game in the bedroom. (A dedicated script for creating it from scratch
is on the list of things to clean up and add back — for now, it can be
created by manually starting a new game in PyBoy and calling
`state.save_state()` at the right moment.)

## Status, honestly

This is an early-stage, actively-changing project. Some scripts overlap or
duplicate each other (visible above) because they were stepping stones
while figuring out exact routes and coordinates, not a finished, polished
pipeline. That's expected and fine at this stage — the priority has been
getting each small fact right (this coordinate, this memory address, this
reward) before building the next thing on top of it.
