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

Everything below exists in this repository today and runs.

### Talking to the emulator

- **`src/emulator.py`** starts PyBoy, points it at your Pokemon Red ROM
  (which you provide yourself — see below), and sets how fast it runs.
  `run_frames()` just advances the emulator a given number of frames —
  Pokemon Red runs at roughly 60 frames per second, so `run_frames(pyboy, 60)`
  is "wait about one second."
- **`src/controls.py`** presses and releases Game Boy buttons. The
  interesting part is `walk_tile()`: instead of holding a direction for a
  fixed amount of time (which sounds simple but isn't — too short and the
  character just turns to face that way without moving, too long and it
  walks multiple tiles), it holds the button and checks the character's
  position *every single frame* until the position actually changes, then
  releases. This was a real bug fixed early on, and it's the reason
  movement in this project is precise instead of flaky.

### Reading the game's memory

- **`src/memory.py`** currently knows three facts: which map you're on
  (`ADDR_MAP_ID`), and your X and Y position on that map. That's enough to
  know "where am I" and "did I just move," which is all the current tasks
  need.

### Remembering where you left off

- **`src/state.py`** saves and loads PyBoy save states — snapshots of the
  entire game at one instant. Right now there's one: `bedroom.state`,
  captured right after starting a new game, standing in the bedroom. This
  means every test can start from "already in the bedroom" in about a
  second, instead of replaying the intro every single time.

### Taking pictures

- **`src/screen.py`** saves a screenshot of the current game screen. Mostly
  used for visually double-checking that a script actually did what the
  memory readings claim it did.

### Turning "leave the house" into something an AI can practice

This is the heart of the project so far — turning a task into something
shaped like a game *for* the AI, with a score to try to maximize:

- **`src/actions.py`** defines what the agent is currently allowed to do:
  move up, down, left, or right. Nothing else yet — no menus, no buttons
  like A or B. Keeping the choices small keeps the learning problem small.
- **`src/rewards.py`** is the scoring rule for the "leave the house" task.
  Every step costs a tiny penalty (encourages shorter paths), standing
  still costs more (discourages walking into walls), visiting a new tile
  for the first time earns a small reward (encourages exploring instead of
  pacing back and forth), reaching the downstairs area earns more, and
  reaching Pallet Town outside — the actual goal — earns a large reward.
- **`src/simple_env.py`** ties it all together into a loop that should look
  familiar if you've heard of reinforcement learning before:
  `reset()` loads the bedroom save state and hands back the starting
  position; `step(action)` performs one move, checks the new position,
  computes the reward using `rewards.py`, and reports back whether the
  episode is finished (either the goal was reached, or too many steps
  passed). Nothing in here is Pokemon-specific logic telling the character
  what to do — it's just the scoreboard and rulebook. Whatever plays
  through this loop is the thing that has to actually figure out the path.

### Proving the game-within-a-game works

- **`src/run_random_agent.py`** plays the above environment by picking
  completely random moves. It isn't meant to be smart — it exists to prove
  the environment itself works correctly (rewards make sense, the episode
  ends when it should, nothing crashes) *before* plugging in something that
  is actually trying to learn. This script is where the project currently
  stands: the scaffolding is proven, and a real learning agent (a
  Q-learning agent — see the roadmap below) is the next thing to build on
  top of it.

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

1. **Actually train the Q-learning agent.** Right now `simple_env.py` is
   proven but nothing intelligent has learned to use it yet. Next up: a
   small **Q-learning agent** — essentially a lookup table that maps "where
   am I" to "which direction has worked out best from here so far,"
   updated after every attempt. It starts knowing nothing and gradually
   gets better purely from trial and error and the reward signal above.
2. **Chain more of the game's opening**: reaching Professor Oak, choosing a
   starter Pokemon, and the forced first battle against your rival — each
   following the same pattern (small scripted probe to learn the exact
   trigger/coordinates, then a small trainable task).
3. **A real turn-based battle is a different kind of problem than walking
   around**, so it gets a different kind of learner:
   - Walking around has a small, easily-listed set of good states (map,
     x, y) — a lookup table handles that fine, and it's easy to inspect
     and debug by hand.
   - Battles involve HP totals, move choices, and outcomes that don't
     compress into a small table the same way — so the battle skill will
     be learned by a small **neural network** (a DQN, trained using the
     Stable-Baselines3 library) instead. Walking keeps its lookup table;
     battling gets a network. Different tool for a different shape of
     problem, not an upgrade for its own sake.
   - Battle actions will be things like "use move 1," not raw button
     mashing through menus — the menu cursor movement itself is
     scaffolding, the same way `walk_tile()` is scaffolding for movement.
     What has to be *learned* is which move to pick, not how to physically
     wiggle a cursor.
   - The battle memory addresses (HP, whether a battle is active, etc.)
     will be sourced from Pokemon Red's long-since publicly documented RAM
     map, then double-checked by hand against this project's own save
     states before being trusted.
4. **A hand-written "controller"** will eventually chain the individually
   trained skills together for a real end-to-end run — run the walking
   skill until a battle starts, then hand control to the battle skill,
   then hand control back — the same philosophy as everything above, just
   one level up: script the handoffs, learn the behavior.
5. **Later still**: wild Pokemon encounters (a different flavor of battle,
   since unlike a rival fight you actually *can* run away), healing
   strategy, and eventually eight badges and eventually the Elite Four —
   each one added only once the step before it is actually working, not
   designed for prematurely.

## Try it yourself

You'll need your own legally-owned Pokemon Red ROM — this project
deliberately never includes or distributes one.

```bash
# 1. Set up your Python environment (once)
python3 -m venv .venv
source .venv/bin/activate
pip install pyboy numpy pillow

# 2. Add your ROM
#    Place it at: roms/pokemon_red.gb

# 3. Watch the environment work with a random (not-yet-smart) agent
python src/run_random_agent.py
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
