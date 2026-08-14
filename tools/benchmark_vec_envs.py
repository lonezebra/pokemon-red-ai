"""
Find out what actually limits whole-game training throughput.

Written because the first real measurement was alarming: one environment ran
at ~650 agent-steps/sec, and fourteen of them ran at ~890 -- a 14x increase
in emulators buying a 1.4x increase in throughput. Something other than
emulation is the constraint, and adding cores (locally or rented) is wasted
money until it's identified.

This separates the three candidates by measuring them apart:

  raw     -- N environments stepped directly, no vectorization at all.
             This is what the CPU can actually do.
  vec     -- the same N behind SubprocVecEnv with random actions, no policy.
             The gap between this and `raw` is pure IPC and synchronisation.
  policy  -- the same again, but asking the real network for each action.
             The gap between this and `vec` is inference.

    cd src && ../.venv/bin/python3 ../tools/benchmark_vec_envs.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import multiprocessing  # noqa: E402

import numpy as np  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from envs.whole_game_env import PokemonRedWholeGameEnv  # noqa: E402

EPISODE_STEPS = 100_000  # never truncate mid-benchmark


def make_env():
    return PokemonRedWholeGameEnv(max_steps=EPISODE_STEPS)


def _env_factory():
    return make_env()


def bench_raw(num_envs, steps):
    """N environments in this one process, stepped in turn. No IPC, no
    policy -- the ceiling everything else is measured against."""
    envs = [make_env() for _ in range(num_envs)]
    try:
        for env in envs:
            env.reset()
        rng = np.random.default_rng(0)

        start = time.monotonic()
        for _ in range(steps):
            for env in envs:
                env.step(rng.integers(0, env.action_space.n))
        elapsed = time.monotonic() - start
    finally:
        for env in envs:
            env.close()

    return steps * num_envs / elapsed


def bench_vec(num_envs, steps, vec_class):
    vec = vec_class([_env_factory for _ in range(num_envs)], **(
        {"start_method": "spawn"} if vec_class is SubprocVecEnv else {}
    ))
    try:
        vec.reset()
        rng = np.random.default_rng(0)

        start = time.monotonic()
        for _ in range(steps):
            vec.step(rng.integers(0, vec.action_space.n, size=num_envs))
        elapsed = time.monotonic() - start
    finally:
        vec.close()

    return steps * num_envs / elapsed


def bench_policy(num_envs, steps):
    from stable_baselines3 import PPO

    vec = SubprocVecEnv(
        [_env_factory for _ in range(num_envs)], start_method="spawn"
    )
    try:
        model = PPO("MultiInputPolicy", vec, n_steps=128, device="cpu",
                    verbose=0)
        obs = vec.reset()

        start = time.monotonic()
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, _, _, _ = vec.step(action)
        elapsed = time.monotonic() - start
    finally:
        vec.close()

    return steps * num_envs / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300,
                        help="steps per env per measurement")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8, 14])
    args = parser.parse_args()

    print("=" * 74)
    print(f"Throughput, agent-steps/sec ({args.steps} steps per env each)")
    print("=" * 74)
    print(f"{'envs':>5} {'raw':>10} {'dummy':>10} {'subproc':>10} "
          f"{'+policy':>10}   {'raw/env':>8}")
    print("-" * 74)

    for num_envs in args.envs:
        raw = bench_raw(num_envs, args.steps)
        dummy = bench_vec(num_envs, args.steps, DummyVecEnv)
        subproc = bench_vec(num_envs, args.steps, SubprocVecEnv)
        policy = bench_policy(num_envs, args.steps)

        print(f"{num_envs:>5} {raw:>10.0f} {dummy:>10.0f} {subproc:>10.0f} "
              f"{policy:>10.0f}   {raw / num_envs:>8.0f}")

    print("-" * 74)
    print("raw     = no vectorization  (what the CPU can do)")
    print("dummy   = DummyVecEnv       (same process, no IPC)")
    print("subproc = SubprocVecEnv     (gap vs raw = IPC + sync cost)")
    print("+policy = SubprocVecEnv + network inference each step")
    print("=" * 74)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
