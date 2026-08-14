"""
Carry the 330M-step whole-game policy across the (x, y) observation change
without throwing away what it learned.

Adding the player's tile coordinates to the stats vector grew it from 9 to
11 features, which changes the shape of exactly one thing in the network:
the first Linear layer of the policy/value MLP, whose input is the CNN's
screen features concatenated with the stats vector. Everything else -- the
whole screen CNN, both MLP hidden layers, the action and value heads -- is
shape-identical and copies across untouched.

The two new columns are zero-initialised on purpose. That makes the warm
-started policy *behaviourally identical* to the 330M checkpoint on its
first step: a zero column contributes nothing to the pre-activation, so the
same observation produces the same action distribution. It is not dead
weight, though -- the gradient with respect to a zero weight is still the
input times the upstream gradient, so the moment (x, y) correlates with
better returns those columns start moving. Random-initialising them instead
would perturb a policy that took 330M steps to train, for no benefit.

Verification is not optional here and is done at the bottom: the script
builds a real observation, feeds its first 9 stats to the old model and all
11 to the new one, and asserts the action logits match. If the column order
or concatenation order were wrong, that check fails loudly rather than
producing a silently rewired policy.

    cd src && ../.venv/bin/python3 ../tools/warm_start_xy_observation.py
"""

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from core.config import PROJECT_ROOT  # noqa: E402
from envs.whole_game_env import PokemonRedWholeGameEnv  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "whole_game_ppo"
SOURCE = PROJECT_ROOT / "models" / "whole_game_milestone_330M.zip"
OLD_STATS = 9
NEW_STATS = 11


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"No pre-change checkpoint at {SOURCE}. That backup is the only "
            "copy of the 330M policy with the 9-feature observation."
        )

    print(f"Loading {SOURCE.name} (9-feature observation)")
    # custom_objects sidesteps SB3 rebuilding the old observation space --
    # we only want the weights out of this file, not a runnable model.
    old_model = PPO.load(SOURCE, device="cpu")
    old_sd = old_model.policy.state_dict()

    env = PokemonRedWholeGameEnv(max_steps=64)
    try:
        obs, _ = env.reset()
        assert obs["stats"].shape == (NEW_STATS,), (
            f"env still reports {obs['stats'].shape} stats -- expected "
            f"{NEW_STATS}. Is the env change applied?"
        )

        print("Building a fresh model on the 11-feature observation")
        new_model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=old_model.n_steps,
            batch_size=old_model.batch_size,
            n_epochs=old_model.n_epochs,
            learning_rate=old_model.learning_rate,
            gamma=old_model.gamma,
            gae_lambda=old_model.gae_lambda,
            ent_coef=old_model.ent_coef,
            clip_range=0.2,
            device="cpu",
            verbose=0,
        )
        new_sd = new_model.policy.state_dict()

        # Which tensors need widening is decided by comparing shapes, not by
        # hard-coding SB3's layer names: it has renamed policy internals
        # across versions before, and a stale name would turn this into a
        # silent no-op that ships a half-copied policy.
        widened = []
        copied = 0
        skipped = []

        for key, new_tensor in new_sd.items():
            if key not in old_sd:
                skipped.append((key, "absent in old checkpoint"))
                continue
            old_tensor = old_sd[key]

            if old_tensor.shape == new_tensor.shape:
                new_sd[key] = old_tensor.clone()
                copied += 1
                continue

            # The only legal difference is a wider input on a 2-D weight.
            if (old_tensor.ndim == 2 and new_tensor.ndim == 2
                    and old_tensor.shape[0] == new_tensor.shape[0]
                    and new_tensor.shape[1] > old_tensor.shape[1]):
                merged = torch.zeros_like(new_tensor)
                merged[:, :old_tensor.shape[1]] = old_tensor
                new_sd[key] = merged
                widened.append(
                    (key, tuple(old_tensor.shape), tuple(new_tensor.shape))
                )
                continue

            raise RuntimeError(
                f"Unexpected shape change on {key}: "
                f"{tuple(old_tensor.shape)} -> {tuple(new_tensor.shape)}. "
                "Refusing to guess how to map it."
            )

        new_model.policy.load_state_dict(new_sd)

        print(f"\n  copied unchanged : {copied} tensors")
        print(f"  widened          : {len(widened)} tensors")
        for key, old_shape, new_shape in widened:
            print(f"      {key}: {old_shape} -> {new_shape} "
                  f"(+{new_shape[1] - old_shape[1]} zero columns)")
        if skipped:
            print(f"  skipped          : {len(skipped)}")
            for key, why in skipped:
                print(f"      {key}: {why}")

        if not widened:
            raise RuntimeError(
                "Nothing was widened -- the observation change does not "
                "appear to have reached the network. Not saving."
            )

        # ---- Verification: the warm-started policy must behave identically.
        #
        # The screen is transposed HWC -> CHW by hand here because both
        # models were built behind SB3's VecTransposeImage wrapper, so their
        # policies expect channels-first; feeding the env's raw HWC frame
        # straight in fails on the first conv.
        print("\nVerifying the warm start is behaviour-preserving...")
        screen_chw = np.transpose(obs["screen"], (2, 0, 1))[None, ...]
        old_obs = {
            "screen": screen_chw,
            "stats": obs["stats"][None, :OLD_STATS],
        }
        new_obs = {
            "screen": screen_chw,
            "stats": obs["stats"][None, :],
        }

        with torch.no_grad():
            old_t = {k: torch.as_tensor(v) for k, v in old_obs.items()}
            new_t = {k: torch.as_tensor(v) for k, v in new_obs.items()}
            old_dist = old_model.policy.get_distribution(old_t)
            new_dist = new_model.policy.get_distribution(new_t)
            old_logits = old_dist.distribution.logits.numpy()
            new_logits = new_dist.distribution.logits.numpy()
            old_value = old_model.policy.predict_values(old_t).numpy()
            new_value = new_model.policy.predict_values(new_t).numpy()

        logit_delta = float(np.abs(old_logits - new_logits).max())
        value_delta = float(np.abs(old_value - new_value).max())
        print(f"  max |action logit difference| : {logit_delta:.3e}")
        print(f"  max |value difference|        : {value_delta:.3e}")

        tolerance = 1e-4
        if logit_delta > tolerance or value_delta > tolerance:
            raise RuntimeError(
                "Warm start changed the policy's output. The stats columns "
                "are probably not the trailing features of the concatenated "
                "vector, so zero-padding on the right is wrong. Not saving."
            )
        print("  OK -- identical policy, now with two learnable columns.")

        # Preserve the step counter so training continues the same timeline.
        new_model.num_timesteps = old_model.num_timesteps
        new_model._num_timesteps_at_start = old_model.num_timesteps
        print(f"\n  carried num_timesteps = {new_model.num_timesteps:,}")

        out = MODEL_DIR / "whole_game_latest.zip"
        tmp = out.with_suffix(".zip.tmp")
        new_model.save(str(tmp))
        os.replace(tmp, out)
        print(f"Saved warm-started model to {out}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
