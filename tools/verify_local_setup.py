"""
Preflight check for running this project on a machine other than the
container it was developed in.

Run it before the first training launch:

    .venv/bin/python3 tools/verify_local_setup.py

Everything it checks is something that has actually gone wrong here, or
that differs between this project's Linux/x86 container and a developer
machine. The point is to fail on a clear message now rather than
part-way into a multi-hour training run.

The single most important check is that envs/forest_env imports. That
one is a canary for the whole artifact set: the forest reward function
reads the survey's edge graph at import time, so a missing or stale
screenshots/forest_map_meta.json surfaces as KeyError: 'edges' before an
episode can start -- which is exactly how a container restart here
turned into an hour of unplanned re-surveying.
"""

import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

# Version-pinned in requirements.txt. Reported rather than enforced:
# a mismatch is worth knowing about (PyBoy in particular has changed
# memory-reading and save-state details between releases, and every save
# state in saves/ was written by the pinned version) but is not
# automatically fatal.
EXPECTED = {
    "pyboy": "2.7.0",
    "numpy": "2.4.6",
    "torch": "2.13.0",
    "stable_baselines3": "2.9.0",
    "gymnasium": "1.3.0",
}

# Artifacts that no amount of local compute can regenerate quickly, and
# what depends on each, so a missing one names its own consequence.
REQUIRED_ARTIFACTS = [
    ("roms/pokemon_red.gb", "the ROM -- supply your own legally-owned copy; never committed"),
    ("saves/leveled.state", "every forest episode resets to this"),
    ("models/trainer_battle_dqn.zip", "resolves the forest's forced trainer battles"),
    ("screenshots/forest_map_meta.json", "the survey's tiles/exits/edge graph; the reward function needs it"),
]

failures = []
warnings = []


def check(label, ok, fail_detail="", ok_detail="", fatal=True):
    """
    Detail is per-outcome on purpose. Printing one shared string for both
    outcomes produces lines like "PASS -- no 'edges' key", which is worse
    than no output at all: a preflight that contradicts itself teaches
    the reader to distrust all of it.
    """
    detail = ok_detail if ok else fail_detail
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        (failures if fatal else warnings).append(f"{label}: {fail_detail}")
    return ok


def installed_version(module_name):
    """
    Distribution metadata, not module.__version__ -- PyBoy doesn't expose
    the latter, so reading it reports every install as "unknown".
    """
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(module_name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def _sysctl(name):
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def core_tiers():
    """
    Every CPU performance tier macOS reports, fastest first, as a list of
    (name, physical_core_count).

    Apple Silicon is heterogeneous, and os.cpu_count() flattens that away.
    It matters here because a training round is a barrier: the driver
    joins every worker before merging, so the round takes as long as its
    *slowest* worker. A worker placed on a much slower core doesn't add
    throughput, it adds a straggler everyone waits on.

    Read as an arbitrary number of tiers rather than the obvious two.
    macOS exposes them as hw.perflevel0..N-1, ordered fastest first, with
    hw.nperflevels giving the count -- and while two (performance plus
    efficiency) has been the common layout, nothing in that interface
    promises it. Hardcoding perflevel0 as "the fast cores" silently
    ignores every tier below the top one, which on a three-tier part
    would throw away a whole class of perfectly usable cores.
    """
    if sys.platform != "darwin":
        return None

    count = _sysctl("hw.nperflevels")
    count = int(count) if count and count.isdigit() else 1

    tiers = []
    for level in range(count):
        physical = _sysctl(f"hw.perflevel{level}.physicalcpu")
        if not (physical and physical.isdigit()):
            continue
        name = _sysctl(f"hw.perflevel{level}.name") or f"perflevel{level}"
        tiers.append((name, int(physical)))

    return tiers or None


print(f"\nPokemon Red AI -- local setup check")
print(f"{platform.system()} {platform.machine()}, Python {platform.python_version()}\n")

print("Interpreter")
check(
    "Python 3.10 or newer",
    sys.version_info >= (3, 10),
    f"found {platform.python_version()}; developed on 3.11",
)

print("\nDependencies")
for module, expected in EXPECTED.items():
    try:
        importlib.import_module(module)
    except ImportError as exc:
        check(f"{module} importable", False, str(exc))
        continue

    found = installed_version(module) or "version unreadable"
    # A CPU/CUDA local version suffix (2.13.0+cu130) is the same upstream
    # release, so compare the base version rather than reporting every
    # platform's build as a mismatch.
    base = found.split("+")[0]
    check(
        f"{module} importable",
        True,
        ok_detail=found if base == expected else f"{found} (pinned: {expected})",
    )
    if base != expected:
        warnings.append(f"{module} is {found}, requirements.txt pins {expected}")

# Reported, not checked. Standalone, torch legitimately defaults to one
# thread per core; only under training does that oversubscribe, and the
# parallel entry points already set OMP_NUM_THREADS=1 for their workers.
# Asserting 1 here would fail on a perfectly healthy machine.
try:
    import torch
    print(f"        note: torch.get_num_threads()={torch.get_num_threads()} in this "
          f"process; training workers are pinned to 1 by the parallel entry points")
    if torch.backends.mps.is_available():
        print("        note: MPS is available but deliberately unused -- the "
              "bottleneck is PyBoy (single-threaded CPU emulation), not the "
              "small DQN policies, so cores beat accelerators here")
except Exception:
    pass

print("\nArtifacts")
for relative, why in REQUIRED_ARTIFACTS:
    path = PROJECT_ROOT / relative
    check(
        relative,
        path.exists(),
        fail_detail=f"missing -- {why}",
        ok_detail=f"{path.stat().st_size:,} bytes" if path.exists() else "",
    )

trainer_states = sorted((PROJECT_ROOT / "saves" / "trainer_battles").glob("trainer_*.state"))
check(
    "saves/trainer_battles/trainer_*.state",
    len(trainer_states) == 6,
    fail_detail=f"found {len(trainer_states)}, expected 6 -- forest_env gates "
                f"its trainer probe to these tiles, so a missing one turns that "
                f"trainer into an unpassable wall",
    ok_detail="6 trainer states",
)

meta_path = PROJECT_ROOT / "screenshots" / "forest_map_meta.json"
if meta_path.exists():
    try:
        meta = json.loads(meta_path.read_text())
        check(
            "forest meta contains its edge graph",
            "edges" in meta,
            fail_detail="no 'edges' key -- this is a version predating the graph "
                        "capture; re-run src/survey_viridian_forest.py",
            ok_detail=f"{len(meta.get('edges', ())):,} edges",
        )
        if "edges" in meta:
            counts = f"{len(meta['tiles'])} tiles / {len(meta['exits'])} exits"
            check(
                "forest meta looks complete",
                len(meta["tiles"]) == 713 and len(meta["exits"]) == 9,
                fail_detail=f"{counts} (expected 713 / 9) -- an incomplete survey; "
                            f"the reward function will still build, but its "
                            f"distances won't cover the whole map",
                ok_detail=counts,
                fatal=False,
            )
    except json.JSONDecodeError as exc:
        check("forest meta parses", False, fail_detail=f"truncated or mid-write: {exc}")

print("\nThe canary")
sys.path.insert(0, str(SRC))
os.chdir(SRC)
try:
    from rewards.forest_rewards import _DISTANCES, _MAX_DISTANCE
    summary = f"{len(_DISTANCES)} tiles reach the goal, max {_MAX_DISTANCE} hops"
    check(
        "rewards.forest_rewards builds its distance map",
        len(_DISTANCES) == 713,
        fail_detail=f"{summary} (expected 713 / 148)",
        ok_detail=summary,
    )
except Exception as exc:
    check(
        "rewards.forest_rewards imports",
        False,
        fail_detail=f"{type(exc).__name__}: {exc}",
    )

try:
    importlib.import_module("envs.forest_env")
    check("envs.forest_env imports (loads the trainer DQN)", True)
except Exception as exc:
    check(
        "envs.forest_env imports (loads the trainer DQN)",
        False,
        fail_detail=f"{type(exc).__name__}: {exc} -- if this is a torch or pickle "
                    f"error, the DQN was saved on Linux/x86 and may need "
                    f"re-training on this machine",
    )

print("\nEmulator")
try:
    from core.emulator import create_emulator, run_frames
    from core.state import load_state
    from core.memory import get_player_position
    from core.config import PROJECT_ROOT as CFG_ROOT

    pyboy = create_emulator()
    load_state(pyboy, CFG_ROOT / "saves" / "leveled.state")
    run_frames(pyboy, 30)
    position = get_player_position(pyboy)
    pyboy.stop()

    where = f"map {position['map_id']} at ({position['x']},{position['y']})"
    check(
        "loads leveled.state and reads the player's position",
        position["map_id"] == 51,
        fail_detail=f"{where}; expected map 51 (Viridian Forest) at (17,47) -- a "
                    f"save state written by a different PyBoy version can load "
                    f"without error but land somewhere wrong",
        ok_detail=f"{where}, Viridian Forest",
    )
except Exception as exc:
    check(
        "emulator boots and loads a save state",
        False,
        fail_detail=f"{type(exc).__name__}: {exc}",
    )

print("\nParallelism")
total = os.cpu_count() or 1
tiers = core_tiers()
print(f"  os.cpu_count() reports {total}")

if tiers:
    for name, count in tiers:
        print(f"    {count:2d} x {name}")

    # All of them. Workers claim episodes from a budget shared across the
    # round rather than each being handed an equal share (see
    # train_navigation_parallel.run_worker), so a slower core simply
    # completes fewer episodes instead of becoming a straggler the round
    # waits on. That removes the reason to exclude a tier at all: an
    # excluded core contributes nothing, whereas an included slow one
    # contributes whatever it manages.
    #
    # Excluding tiers was in fact the wrong call. A rule of "drop the
    # slowest tier" costs almost nothing where that tier is a handful of
    # efficiency cores, but on a 6-super/12-performance split it would
    # discard two thirds of the machine -- both of those tiers are fast,
    # and the queue makes the difference between them a non-issue.
    recommended = max(1, sum(count for _, count in tiers))
    print(f"  -> Recommended:  POKEMON_RED_WORKERS={recommended}  (all physical cores)")
    if len(tiers) > 1:
        print(f"     Uneven tiers are handled by the work queue rather than by")
        print(f"     excluding cores: each worker claims episodes until the round's")
        print(f"     budget runs out, so faster cores do proportionally more and")
        print(f"     every worker finishes together. The per-worker spread printed")
        print(f"     each round shows this happening.")
else:
    recommended = total
    print(f"  -> Recommended:  POKEMON_RED_WORKERS={recommended}")

print(f"  each worker measured ~700MB resident here, so {recommended} workers "
      f"needs roughly {recommended * 0.7:.1f}GB")

try:
    import train_navigation_parallel  # already on sys.path via SRC
    budget = train_navigation_parallel.DEFAULT_EPISODES_PER_ROUND
    print(f"  {budget} episodes per round, shared across all workers "
          f"(~{budget / recommended:.0f} each if cores were identical)")
    print(f"  (the budget is the round total, so more cores shorten rounds rather")
    print(f"   than enlarging them -- POKEMON_RED_EPISODES_PER_ROUND overrides it)")
except Exception:
    pass

print()
print("  Sharing the machine: every worker pins one core at ~100%, so leave")
print("  headroom if you want the machine usable -- POKEMON_RED_WORKERS is read")
print("  at launch, and training resumes from its last completed round, so")
print("  stopping a run and relaunching with fewer workers loses nothing.")

print()
if failures:
    print(f"{len(failures)} blocking problem(s):")
    for failure in failures:
        print(f"  - {failure}")
if warnings:
    print(f"{len(warnings)} thing(s) worth knowing:")
    for warning in warnings:
        print(f"  - {warning}")
if not failures:
    print("Ready. To train:")
    print(f"  cd src && POKEMON_RED_WORKERS={recommended} python3 train_forest_agent.py")
    print("And to keep artifacts safe across restarts, in a second shell:")
    print("  tools/checkpoint_artifacts.sh")

sys.exit(1 if failures else 0)
