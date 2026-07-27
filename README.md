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
  house, arriving at Oak's Lab, having just picked a starter Pokemon, the
  moment the first rival battle begins, and standing in Route 1's tall
  grass just after winning that battle. Resetting to any of these takes
  about a second, instead of replaying everything before it.
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

`battle_env.py` also randomizes the player's starting battle stats on
every reset by default (`randomize_stats=True`), rolling them within the
real range a freshly-obtained level-5 Squirtle can have (see
`memory.randomize_battle_mon_stats`). This isn't a hypothetical concern —
seeing it matter is exactly why it's here. **Current results: 92/100
wins evaluated against randomized stats (0 losses turned into stuck
episodes — see below), and 50/50 wins against the original fixed
matchup** the project started with — so the trained policy generalizes
in both directions rather than trading one for the other. The project's
bar for "good enough to move on" was 90/100.

Three real bugs were caught and fixed while building this, all by
testing against known ground truth instead of trusting the first version
that ran without crashing:

- The battle menu's move cursor turned out to be **sticky** (it
  remembers the last move you used, rather than resetting each turn) and
  the move list **wraps around** instead of stopping at the top/bottom.
  A script that just pressed "up" a few times to "reset" the cursor was
  actually landing on the wrong move most of the time — caught by
  checking which move's PP actually went down, not just trusting the
  intended button sequence.
- See `advance_battle_dialogue()` above for the second one.
- Picking an invalid move slot originally left the game state completely
  unchanged (just a penalty, no button pressed). That seemed harmless
  until a trained policy's greedy choice for some observation happened
  to *be* an invalid slot — since nothing about an unchanged state ever
  looks different to the network, it would deadlock forever repeating
  that exact wrong pick, burning every remaining step of the episode.
  Caught by noticing evaluation episodes were timing out frozen at the
  same HP values instead of ending in a win or loss. Fixed by having the
  environment substitute the first valid move and actually play it —
  still penalized, but never able to stall the battle completely.

### Chaining trained skills together (`src/agents/skills.py`, `src/controller.py`)

This is the payoff the whole project has been building toward: **one
continuous run**, from waking up in the bedroom all the way to beating
the rival and stepping into Route 1, with no manual save-state hand-offs
in the middle.

```
bedroom.state -> [leave-house Q-agent] -> Pallet Town
              -> [scripted route]      -> Oak -> lab -> choose Squirtle
              -> [scripted route]      -> rival's trigger
              -> [rival-battle DQN]    -> win
              -> [scripted route]      -> Route 1's entrance
```

- **`src/agents/skills.py`** wraps each trained skill behind the exact
  same interface: `skill.choose_action(observation)`. Under the hood,
  `LeaveHouseSkill` is a dictionary lookup and `RivalBattleSkill` calls a
  neural network's `.predict()` — `controller.py` calling
  `choose_action()` never needs to know or care which.
- **`src/controller.py`** runs the whole sequence above over a single
  shared PyBoy session, mixing learned skills and scripted routes freely
  since they all speak the same "read state, decide, act" language.

Getting the hand-off from segment 1 to segment 2 actually working needed
one more real fix, not just calling things in the right order. The
leave-house Q-agent's episode ends the instant `map_id` becomes 0
(Pallet Town) — but that's not the final resting position. The game
keeps auto-walking the player a couple more tiles out of the doorway on
its own afterward, no input needed. Trying to act immediately (an
earlier version of this controller did) collided with that in-progress
automatic movement and produced a nonsensical multi-tile position jump.
The fix, `controls.wait_for_position_to_settle()`, just polls position
until it stops changing on its own — the same "check the real state,
don't guess" idea as everywhere else in this project — and reliably
lands at the exact tile `saves/outside_house.state` represents, which is
where the scripted route to Oak's trigger already assumes it starts.
Verified reliable across multiple full runs, bedroom to battle won.

### Reaching Route 1 (`src/create_route1_entry_state.py`)

Winning the rival battle isn't the end of the road — the controller now
carries on one segment further, out of Oak's Lab and up to the edge of
Route 1, the path toward Viridian City and, eventually, Brock.

Finding this route needed real investigation, not just chaining known
pieces together, because two assumptions turned out to be wrong:

- Winning the battle doesn't hand control back immediately — there's a
  post-battle dialogue sequence to clear first (Blue reacting to the
  loss), the same "press A, then test for real movement, stop the
  instant it works" pattern used everywhere else in this project for
  dialogue-then-movement handoffs.
- Oak's Lab has its *own* exit door, separate from the player's house.
  Walking out of it does not land you back at `outside_house.state`'s
  position — it's a different spot in Pallet Town entirely, since it's a
  different building. `wait_for_position_to_settle()` (already built for
  the player's-house exit) was needed here again, for the same reason:
  the game keeps auto-walking the player a couple more tiles on its own
  right after the warp.
- The gap in the hedge that actually leads to Route 1 is *not* a straight
  line north from the lab's door — that column is blocked almost
  immediately. The real gap sits further west (directly above Professor
  Oak's own "Hey! Wait!" trigger tile from the starter-selection route),
  and reaching it means walking *around* the lab building first. This was
  found the same way every other coordinate in this project was found:
  systematically testing which tiles allow movement in which direction,
  one probe at a time, rather than guessing from a mental picture of the
  map.

The verified route is a straight sequence once known — right 4, up 10,
left 6, up 3 — reliably taking the player from just outside the lab's
door to standing in Route 1's tall grass (confirmed against the public
[pret/pokered](https://github.com/pret/pokered) map-ID constants: map 12
is Route 1). `src/controller.py`'s new **Segment 4** plays this out live
after the rival battle is won, and `create_route1_entry_state.py` saves
the result as `saves/route_1_entry.state`, the checkpoint the next
milestone (actual Route 1 navigation) will start from.

### Scripting the missing middle route (`src/create_starter_obtained_state.py`)

This closes the gap the controller and an earlier roadmap version both
called out: walking from Pallet Town to Professor Oak's trigger, through
the automatic walk-in to the lab, and choosing Squirtle, all from
`outside_house.state` — no pre-made save state required for this part
anymore.

Two of the same "don't trust a fixed press count" bugs as elsewhere in
this project showed up again here, in new forms:

- Part of this sequence is genuinely automatic (the game itself walks
  the player from the door to a fixed arrival tile) — but the exact
  number of A-presses needed to get *through* the dialogue before and
  after it drifts slightly between runs, because a few frames of timing
  variance earlier (from `walk_tile`'s "hold until moved" checks) shift
  where later dialogue pagination lands. Worse, once real control
  actually returns, keep pressing A the fixed number of times and it just
  re-triggers the same prompt again. The fix: press A once, immediately
  test the actual next move in the real route, and only press A again if
  that test fails — so the loop stops the instant control is genuinely
  back, never overshooting.
- Detecting "the starter was actually obtained" needed its own real
  signal, the same way battle-end detection did: `wPartyCount` (0xD163),
  verified to read 0 before a starter and 1 after across every save state
  in this project before being trusted.

**An important limitation this surfaced, not papered over:** the
starter's hidden stats (IVs) are randomly rolled at the moment it's
created, based on the game's RNG state — which depends on the exact
number of frames consumed by everything before that point. Running this
script doesn't reproduce the *exact same* Squirtle every time, just *a*
Squirtle at the same milestone. Checked directly: a fresh run produced a
Squirtle with 19 max HP instead of 20, and feeding that into the
then-trained battle DQN dropped it from 100/100 wins to 0/10 — the
policy had implicitly learned the specifics of one exact matchup, not
"Squirtle vs. this Bulbasaur" in general.

**Update: fixed.** Rather than trying to reproduce one exact Squirtle,
`battle_env.py` now randomizes the player's stats within the real
possible range on every reset (see the battle section above) and the
model was retrained against that. Generating several sample starters and
reading their actual stats (rather than trusting a stat formula from
memory, which turned out to be slightly wrong) also showed the enemy's
stats never vary at all — Gen 1 trainer Pokemon have fixed IVs — so only
the player's side needed this.

### Route 1 navigation (`src/rewards/route1_rewards.py`, `src/envs/route1_env.py`, `src/train_route1_agent.py`, `src/train_route1_agent_parallel.py`)

Teaching an agent to actually walk Route 1 to Viridian City, rather than
following a scripted route — the same tabular Q-learning approach as the
leave-house task, since the state space (`map_id`, `x`, `y`) is still just
three numbers, but complicated by wild Pokemon encounters interrupting
movement at random.

- `envs/route1_env.py` handles wild encounters automatically (always
  attempts to run — unlike the rival fight, running from a wild Pokemon is
  always legal) so the task the agent actually has to learn stays just
  navigation. Battling wild Pokemon is its own later milestone, not this
  one's.
- Two real problems surfaced only by actually evaluating the trained
  policy, not trusting the training-time success counter:
  - **The reward function accidentally rewarded farming Pallet Town.** The
    "+1 for a new tile" bonus didn't check *which* map the new tile was
    on, so a trained agent's very first greedy move turned out to warp
    backward into Pallet Town and just stay there, earning the same
    novelty reward exploring Pallet Town as it would for real Route 1
    progress — an easier way to rack up reward than pushing toward a goal
    it had never once reached. Fixed two ways: `route1_env.py` now ends
    the episode the instant the player leaves Route 1 backward, and the
    reward function gives an explicit -20 penalty for that outcome
    instead of just withholding the bonus.
  - **Even after that fix, the trained policy converged to a
    directionless revisit loop, not real progress.** Confirmed directly:
    a full greedy playthrough ran the entire 800-step budget but visited
    only 24 distinct tiles, spending 777 of 801 steps revisiting ground
    already covered. The deeper cause: the per-episode novelty bonus
    depended on each *episode's own* visitation history
    (`visited_positions` resets every `reset()`), so the same (state,
    action) pair could be rewarded in one training episode and not
    another — a noisy, inconsistent target for tabular Q-learning, which
    assumes reward is a function of state alone. Replaced with
    **potential-based reward shaping** (Ng, Harada & Russell 1999):
    `route1_potential(position) = -y` on Route 1 (a mostly-vertical
    corridor — y=35 at the entrance, y=0 at Viridian City's end), a pure
    function of position with no episode-history dependence. Moving away
    from the goal is now an explicit penalty, not just a forfeited bonus.
- **Current result: ~96-99% success rate**, converging within about 15
  rounds of training and staying stable afterward — unlike the earlier
  reward scheme, which never once reached the goal across thousands of
  episodes, and whose apparent exploration progress would regress rather
  than hold. The learned policy solves the route in as few as ~53 steps,
  well under the ~670-step reference from an early up-biased random-walk
  scout (that scout was never trying to be efficient, just thorough).
- **`src/train_route1_agent_parallel.py`**: this container has 4 CPU
  cores, and PyBoy training is CPU-bound C code that threads can't help
  with past the GIL. This runs training as independent worker processes
  instead — each with its own emulator instance, all starting from the
  same shared Q-table each round, training independently, then merged
  back together by averaging every (state, action) value the workers
  actually have an opinion on. Roughly 4x the throughput of the
  single-process version, with the same episode/checkpoint crash-recovery
  guarantee.
- **`src/build_route1_map.py`, `src/generate_route1_mashup_rollouts.py`,
  `src/render_route1_mashup.py`**: a "run mashup" visualization — stitches
  a real panorama of Route 1 from actual screen captures (median-stacked
  across many overlapping frames to erase the moving player sprite,
  leaving just the static terrain), then overlays many independent
  rollouts of the trained agent moving simultaneously across it, colored
  by outcome. Mostly a fun/diagnostic side project, but it's what actually
  surfaced the revisit-loop bug above in the first place — 150 "greedy"
  rollouts against the first fully-trained checkpoint came back
  bit-for-bit identical, which is what prompted recording one in detail
  rather than trusting the aggregate numbers.

### Wild Pokemon encounters (`src/rewards/wild_battle_rewards.py`, `src/envs/wild_battle_env.py`, `src/train_wild_battle_agent.py`, `src/watch_wild_battle_agent.py`)

A different flavor of battle from the rival fight: the opponent's
species varies, and unlike a rival fight, running away is actually
legal — so this got its own environment rather than a mode of
`battle_env.py`.

- Two facts nothing in this project needed to read before: the enemy's
  and player's species and level (`core/memory.py`). Verified directly
  against a live Route 1 encounter before being trusted — walked into
  the grass, advanced to the FIGHT/PKMN/ITEM/RUN menu, and cross-checked
  the read values against the actual on-screen "PIDGEY :L3" text. Gen 1
  stores species as an internal index, not the Pokedex number — reading
  36 there and independently knowing Gen 1's internal index lists Pidgey
  at 36 confirmed both the address and that assumption at once.
- **`src/create_wild_encounter_state.py`** captures a training save state
  for each *distinct* wild species Route 1 can produce, rather than one
  fixed encounter — found both of the only two it has (Pidgey, Rattata).
  `wild_battle_env.py` resets from one of these at random each episode,
  the same opponent-variety problem this milestone exists to solve.
- The environment adds one real new thing beyond the rival battle env's
  design: a RUN action, attempted once per environment step rather than
  auto-retried — a failed attempt is visible in the next observation, so
  the agent decides for itself whether to try again or fight instead,
  rather than a scripted retry loop deciding on its behalf. The reward
  function correspondingly handles the battle ending with neither side's
  HP at zero (the player fled, or the wild Pokemon fled on its own —
  treated the same), rewarded less than a win so winning stays preferred
  whenever it's actually achievable.
- **Current result: 100/100 wins, 0 losses, 0 fled**, evaluated the same
  way as the rival battle (100 greedy episodes). Route 1's actual wild
  encounters (Pidgey/Rattata, level 2-3) are trivially weak against a
  level 6 Squirtle, so this result mainly proves the infrastructure
  (opponent variety, the run action, the reward shape) rather than
  fight-or-flee judgment specifically — the agent never needed to flee
  to win every time. That's an honest limitation, not a hidden one:
  proving it can flee *well* needs a wild Pokemon actually worth
  avoiding, which Route 1 alone doesn't provide.

### Route 2 navigation (`src/rewards/route2_rewards.py`, `src/envs/route2_env.py`, `src/train_route2_agent.py`, `src/watch_route2_agent.py`)

The same kind of task as Route 1, and deliberately the first one where
both ends were established *before* any training ran — because the
previous attempt at "Route 2" was trained for 1500 episodes against a
route that had no reachable goal at all (see the section above).

`core.pathfind.survey_map()` flood-filled Route 2 and finished with its
frontier exhausted at 244 tiles, so this exit list is complete rather
than whatever a random walk happened to stumble into:

| From | Direction | Leads to |
|------|-----------|----------|
| (7,71) (8,71) (9,71) | down | map 1 — back to Viridian City |
| **(3,44)** | **up** | **map 50 — Viridian Forest south gate** |

Exactly one forward exit, which settled two things at once. `reached_goal`
could name map 50 specifically, rather than the previous task's vague
"any map that isn't this one" (which had counted walking through a
building door as a win). And entry at y=71 against a goal at y=44 fixed
*which direction is forward* — decreasing y, the same orientation as
Route 1 — so Route 1's `-y` potential shaping carried over unchanged, as
a verified fact rather than an assumption.

**Current result: 50/50 in evaluation, taking exactly 41 steps every
time** — a fully deterministic path, against a theoretical minimum of 27
tiles of vertical progress. It converged on the first attempt, reaching
98%+ per-round success by round 15 and holding there, with the greedy
policy solving the route from round 1 onward. The contrast with the
Route 22 attempt is the entire lesson: identical algorithm, identical
reward shaping, and the only thing that changed was checking the goal
was reachable first.

Also adds **`src/train_navigation_parallel.py`**, a route-agnostic
version of the Route 1 parallel trainer — environment, model paths and
GIF names are parameters — so Viridian Forest and whatever follows don't
each need another copy of the loop.

### Oak's Parcel, and the road north (`src/core/pathfind.py`, `src/create_pokedex_obtained_state.py`, `src/create_route2_entry_state.py`)

The most instructive mistake in this project so far, and the tooling
built to make sure it doesn't happen again.

**The mistake.** After Route 1 was solved, the next milestone was Route
2. A biased random walk out of Viridian City reached a map, a screenshot
confirmed it was real outdoor route terrain rather than a building
interior, and it was labelled Route 2. A full navigation task was built
on it and trained for 1500 episodes, which produced essentially nothing
— exploration would improve for a while, then collapse. That looked
exactly like the reward-shaping instability Route 1 had suffered, so the
natural assumption was that it needed the same fix.

It didn't. The map was **Route 22**, west of Viridian, which dead-ends
at the Victory Road gate and its eight-badge check. The task had no
reachable goal at all, so no reward function would ever have fixed it.
The screenshot that "confirmed" the map only ever established it was *a*
route, never *which* route — it felt like verification without testing
the actual claim. Three checks settled it afterwards: exiting that map
eastward lands at Viridian's far *west* edge (x=0); the map-ID table,
anchored on two IDs this project had already verified independently
(Oak's Lab 40, Route 1 12), puts Route 22 at 33; and its west end is a
solid mountain wall.

**The real blocker.** Viridian City's northern exit is closed until Oak's
Parcel is delivered, and this project's controller skips that errand
entirely (bedroom → lab → starter → rival → Route 1). Measured directly
by flood-filling the city before and after running it:

|        | reachable tiles | y range | exits north |
|--------|-----------------|---------|-------------|
| before | 500             | 4 – 35  | none        |
| after  | 600             | 0 – 35  | (17,0) (18,0) (19,0) → map 13 |

`create_pokedex_obtained_state.py` runs that errand — Mart, Parcel, back
through Route 1 to Pallet Town, Oak, Pokédex — and
`create_route2_entry_state.py` then reaches the genuine Route 2 (map 13,
entered at its southern end around (7–9, 71)). Both track progress by
reading the game rather than counting button presses: the Parcel is
followed through the bag in memory, verified empty outside the Mart, one
entry the moment the clerk hands it over, and empty again the moment Oak
accepts it.

- **`src/core/pathfind.py`** is the tooling half. Every route before this
  was found by biased random walks with stuck-escape heuristics, which
  are slow, seed-dependent, and — the real problem — cannot distinguish
  "there is no path" from "the walk got unlucky". That ambiguity is
  precisely what hid the Route 22 mistake for so long. This does a
  breadth-first search over real save states instead, so exhausting the
  frontier is a genuine proof of absence; it is what established that
  Viridian had no northern exit at all. On success it *loads the
  snapshot it found* rather than replaying the moves that got there,
  because replaying can diverge — Route 1's grass interrupts steps with
  wild encounters, and fleeing takes a different number of turns each
  time.

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
   **Done** — `create_starter_obtained_state.py` scripts the whole route
   from `outside_house.state`, and `create_rival_battle_state.py`
   reliably takes it the rest of the way. The IV-variance caveat this
   used to carry is resolved -- see item 3.
3. ~~Battles need a different kind of learner than walking does.~~ **Done**
   — see `battle_env.py`/`train_battle_agent.py` above. The rival battle
   is beaten reliably (92/100 against randomized starting stats, 50/50
   against the original fixed matchup) by a small neural network, not a
   lookup table, matching the reasoning that HP totals and move outcomes
   don't compress into a small table the way three coordinates do. The
   policy is also now robust to the starter IV variance from item 2,
   rather than having implicitly memorized one exact matchup.
4. ~~A hand-written "controller" chains individually trained skills
   together.~~ **Done** — see `src/controller.py` above. One continuous
   run, `bedroom.state` to a won rival battle, verified reliable across
   multiple runs. (An earlier attempt at this hit what looked like a
   real blocker — the leave-house Q-agent's path exits the house at a
   different tile than the scripted route to Oak's trigger assumed — but
   it turned out to be a timing issue, not a layout one: waiting for the
   game's own automatic "step out of the doorway" animation to finish,
   instead of acting immediately, landed at the expected tile every
   time.)
5. ~~Bridge from the won rival battle to Route 1's entrance.~~ **Done** —
   see "Reaching Route 1" above. The controller now runs a fourth
   segment, `bedroom.state` all the way to standing in Route 1's tall
   grass, verified reliable across multiple runs.
6. ~~Route 1 navigation: teach an agent to actually walk Route 1 toward
   Viridian City, rather than a scripted route.~~ **Done** — see "Route 1
   navigation" above. ~96-99% success rate, solving the route in as few
   as ~53 steps.
7. ~~Wild Pokemon encounters: a different flavor of battle from the rival
   fight, since the opponent varies and running away is legal.~~ **Done**
   — see "Wild Pokemon encounters" above. 100/100 wins evaluated against
   Route 1's actual encounters, though genuine fight-or-flee judgment
   isn't proven yet since nothing on Route 1 is worth fleeing from.
8. ~~Unblock the road north out of Viridian City.~~ **Done** — see
   "Oak's Parcel, and the road north" above. Route 2 is now reachable
   and `saves/route2_entry.state` exists.
9. ~~Route 2 navigation, this time from a properly verified
   checkpoint.~~ **Done** — see "Route 2 navigation" above. **50/50** in
   evaluation, a 41-step path every time.
10. **Viridian Forest is surveyed but blocked, and the survey is why we
    know that.** Map 51, reached through the gate at map 50 (both now
    verified rather than inferred from the map table). `survey_map()`
    flood-filled it to completion: **676 reachable tiles**, filling only
    **45% of its bounding box** — so it is a genuine maze, confirming
    the y-coordinate shaping that carried Route 1 over to Route 2 would
    *not* have transferred. But the more important result is that its
    **only reachable exit leads back the way we came**. There is no way
    north to Pewter City from where the agent currently stands.

    That is the Route 22 situation exactly — a task whose goal cannot be
    reached — except this time it cost about twenty minutes of surveying
    instead of 1500 training episodes. Checked hard before trusting it,
    since the negative result is the whole claim: the survey completed
    with its frontier exhausted (not truncated), zero unfleeable battles
    corrupted it, the entire northern tree line was confirmed solid both
    visually and by interacting north from eight separate positions, and
    400 steps of ordinary walking produced 9 wild battles and no trainer
    battles.

    What blocks it looks like the forest's trainers. Probing every
    blocked direction found 72 that respond, several of them Bug
    Catchers ("Hey, wait up!") standing at chokepoints. A trainer
    occupies its tile, so pathfinding correctly reads it as a wall —
    which means getting past them needs the ability to *win* a trainer
    battle, since unlike a wild Pokemon a trainer cannot be fled.
11. **Trainer battles** are therefore the real next milestone, and they
    unblock two things at once: Viridian Forest, and Brock. The
    rival-battle DQN already solves a fixed, unfleeable trainer fight,
    so the work is generalizing it to varying opponents rather than
    starting over.
12. **Later still**: healing strategy (when to retreat/heal rather than
    push through a fight), Pewter City, a new battle environment
    trained specifically for Brock's Rock-type Pokemon, and eventually
    eight badges and the Elite Four — each one added only once the step
    before it is actually working, not designed for prematurely.

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

# 7. Play the trained DQN through the rival battle and on to Route 1's
#    entrance, saving that checkpoint (needs models/rival_battle_dqn.zip
#    from step 5)
python src/create_route1_entry_state.py

# 8. Or run the whole thing in one continuous session, bedroom to Route 1
python src/controller.py
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
