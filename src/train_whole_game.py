import os

# Headless, and set before any core import -- core/config.py reads this at
# import time. Doubly required here: SubprocVecEnv spawns fresh interpreters
# that re-import this module, so a window mode set any later than this would
# open one real Game Boy window per worker.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

# PyBoy is the bottleneck and it is single-threaded; letting torch/BLAS grab
# extra threads inside each worker just oversubscribes the cores the
# emulators need. Same reasoning as train_navigation_parallel.py.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import multiprocessing  # noqa: E402
from pathlib import Path  # noqa: E402

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from core.config import PROJECT_ROOT  # noqa: E402
from envs.whole_game_env import DEFAULT_MAX_STEPS, PokemonRedWholeGameEnv  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "whole_game_ppo"
TENSORBOARD_DIR = PROJECT_ROOT / "models" / "whole_game_tensorboard"

# Leave a few cores for the machine to stay usable -- this run is measured in
# hours, not minutes, and a box you can't type on during it is a box you turn
# off. POKEMON_RED_ENVS overrides.
DEFAULT_NUM_ENVS = max(1, (os.cpu_count() or 4) - 4)
NUM_ENVS = int(os.environ.get("POKEMON_RED_ENVS") or DEFAULT_NUM_ENVS)

TOTAL_STEPS = int(os.environ.get("POKEMON_RED_TOTAL_STEPS") or 10_000_000)

# The GPU is for the gradient updates, not the emulators -- PyBoy is CPU
# work and nothing moves that. It still roughly doubles end-to-end
# throughput, which was not obvious in advance and is worth recording:
# measured on an M5 Max at 14 envs, 746 steps/sec on cpu against 1424 on
# mps. The reason the win is that large is that rollout collection already
# parallelises well (tools/benchmark_vec_envs.py measures ~4600 steps/sec of
# pure stepping at 14 envs), so once the envs are fast the backprop through
# the screen CNN is what's left holding everything up.
DEVICE = os.environ.get("POKEMON_RED_DEVICE") or "mps"

# Steps each env collects per policy update. NUM_ENVS * N_STEPS is the batch
# PPO learns from; 512 x ~14 envs keeps updates frequent enough to see early
# progress without the rollout buffer growing large enough to matter.
N_STEPS = int(os.environ.get("POKEMON_RED_N_STEPS") or 512)

# Episodes are long and the run is long, so checkpoints are frequent. Every
# other training script in this project saves once at the end, which is fine
# for a twenty-minute DQN run and useless for a multi-hour one.
CHECKPOINT_EVERY = int(os.environ.get("POKEMON_RED_CHECKPOINT_EVERY") or 100_000)

START_STATE = os.environ.get("POKEMON_RED_START_STATE")
MAX_STEPS = int(os.environ.get("POKEMON_RED_EPISODE_STEPS") or DEFAULT_MAX_STEPS)

# The collapse-detection gate (see DeliveryGateCallback). Every 5M steps by
# default -- frequent enough to stop a bad run within tens of minutes rather
# than hours, infrequent enough that its short single-env rollouts (a few
# seconds each) are noise against a run measured in hours. 0 disables it,
# for the cases that don't have a milestone-item signal to gate on.
DELIVERY_GATE_EVERY = int(os.environ.get("POKEMON_RED_GATE_EVERY") or 5_000_000)


def atomic_model_save(model, path):
    """model.save(path), but a torn write can never leave a corrupt file at
    `path` for something else to pick up mid-write.

    Stable-Baselines3's save() writes the zip directly at its target path --
    there is no temp-file-then-rename step, unlike core/atomic_io.py's
    write_json_atomic, which this project already built for exactly this
    reason (see its own comment about json.dump not being atomic). A run
    that gets killed mid-save -- Ctrl-C, OOM, a crash, or simply a shell
    command hitting its timeout -- leaves a half-written zip sitting at a
    name both resolve_model() and latest_checkpoint() will confidently treat
    as valid. This is not hypothetical: the first real test of
    tools/monitor_whole_game.py against a checkpoint from this training
    script's own early runs hit `zipfile.BadZipFile: File is not a zip file`
    on exactly this file.

    SB3 only appends ".zip" when the given path has no suffix at all
    (save_util.open_path_pathlib), so a ".zip.tmp" path passes through
    unmodified -- confirmed by reading that function rather than assumed.
    """
    path = Path(path)
    final_path = path if path.suffix == ".zip" else path.with_suffix(".zip")
    tmp_path = final_path.with_suffix(".zip.tmp")

    model.save(str(tmp_path))
    os.replace(tmp_path, final_path)


class AtomicCheckpointCallback(BaseCallback):
    """SB3's CheckpointCallback, but through atomic_model_save.

    Same save_freq/save_path/name_prefix convention (whole_game_<n>_steps.zip
    every save_freq calls), reimplemented rather than wrapped because
    CheckpointCallback has no hook to intercept how the save itself happens.
    """

    def __init__(self, save_freq, save_path, name_prefix="model", verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix

    def _init_callback(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = (
                self.save_path
                / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            atomic_model_save(self.model, checkpoint_path)
            if self.verbose >= 1:
                print(f"Saved checkpoint to {checkpoint_path}")
        return True


class DeliveryGateCallback(BaseCallback):
    """Catch a training collapse in minutes, not the five hours it has cost
    every time so far.

    Three of the last five long continuations from a converged policy have
    collapsed -- most damningly the 350M -> 380M -> 410M sequence, where
    380M reached Viridian Forest and 410M, resumed from it with nothing
    changed but more steps, lost the errand entirely. Recovering from that
    meant noticing late (ep_rew_mean stayed misleadingly healthy -- one
    bisection found delivery already at 0/3 while the curve still read
    ~248) and then manually bisecting hundreds of checkpoints to find the
    last good one. This callback is that bisection, running automatically
    during training instead of after the fact.

    Every `eval_freq` calls it plays a handful of short episodes with the
    *current* policy and counts deliveries -- the same ground-truth signal
    every eval script in this project already uses, not a proxy off the
    training curve. A healthy result promotes the current weights to
    known_good.zip, a separate file from the checkpoint history that only
    ever moves forward when eval confirms the policy still works. Two
    consecutive bad results (not one -- a single unlucky small-sample
    batch happens even to a healthy policy) stop training outright, so a
    collapse burns tens of minutes of compute instead of the full run, and
    known_good.zip is left as the exact resume point.

    Deliberately not integrated into the vectorized training envs: it
    builds one throwaway single-process env per check, so eval never
    perturbs the 14 parallel workers' state or their place in the rollout.

    max_steps defaults to 8192, matching this project's standard eval
    length (check_250m_delivery.py, post_delivery_analysis.py) rather than
    something shorter and cheaper -- not a free choice. Delivery timing
    drifts later as training progresses (34% into the episode at 350M,
    64% at 380M in measured batches), so a gate episode has to be at least
    as long as the standard eval or it stops seeing delivery on a policy
    that is actually fine, and "fine" tips into "flagged as collapsed."
    A smoke test of this callback against the known-good 380M checkpoint
    at max_steps=4096 confirmed exactly that failure before this default
    was corrected -- 0/3 on a policy that delivers 9/12 to 15/16 at 8192.

    episodes defaults to 12 -- this project's standard eval batch size
    (check_250m_delivery.py, etc.) -- for the same reason max_steps got
    fixed: the first real run under this callback stopped itself at 390M
    on two consecutive 0/3 checks, and a follow-up 12-episode eval on that
    exact checkpoint came back 5/12, a perfectly normal rate for this
    policy. At a true per-episode delivery rate of ~42%, 0/3 happens
    ~20% of the time by chance alone, so two in a row (the patience=2
    trigger) was a ~4% roll, not a signal -- and with several check
    windows in a 30M-step run, a coin that unfair will come up more often
    than the callback's whole purpose can tolerate. At 12 episodes the
    same math drops to ~0.15% per check, ~0.0002% for two in a row:
    exact zero (what every confirmed real collapse has actually shown --
    350M's tourist run was 0/28, 410M was 0/12 and 0/6) stays a strong
    signal, and ordinary variance stops looking like one.
    """

    def __init__(self, save_path, eval_freq, episodes=12, max_steps=8192,
                 min_deliveries=1, patience=2, verbose=1):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.max_steps = max_steps
        self.min_deliveries = min_deliveries
        self.patience = patience
        self._consecutive_failures = 0

    def _init_callback(self):
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _run_eval(self):
        # Local import: PokemonRedWholeGameEnv is already imported at module
        # level for the vectorized workers, but constructing it here, once
        # per gate check rather than once per process at import time, makes
        # the throwaway nature of this env explicit at the call site.
        env = PokemonRedWholeGameEnv(max_steps=self.max_steps)
        delivered = 0
        try:
            for _ in range(self.episodes):
                obs, _ = env.reset()
                got_it = False
                for _ in range(self.max_steps):
                    action, _ = self.model.predict(obs, deterministic=False)
                    obs, _, terminated, truncated, info = env.step(action)
                    if info["reward_components"].get("milestone", 0.0) >= 100.0:
                        got_it = True
                    if terminated or truncated:
                        break
                delivered += int(got_it)
        finally:
            env.close()
        return delivered

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        delivered = self._run_eval()
        healthy = delivered >= self.min_deliveries

        if healthy:
            self._consecutive_failures = 0
            known_good = self.save_path / "whole_game_known_good.zip"
            atomic_model_save(self.model, known_good)
            if self.verbose >= 1:
                print(
                    f"[gate] step {self.num_timesteps:,}: "
                    f"{delivered}/{self.episodes} delivered -- OK, "
                    f"known_good.zip updated"
                )
            return True

        self._consecutive_failures += 1
        print(
            f"[gate] step {self.num_timesteps:,}: "
            f"{delivered}/{self.episodes} delivered -- "
            f"BELOW THRESHOLD ({self._consecutive_failures}/{self.patience})"
        )
        if self._consecutive_failures < self.patience:
            return True

        print(
            f"\n[gate] stopping: {self.patience} consecutive checks below "
            f"threshold at step {self.num_timesteps:,}. "
            f"Resume from {self.save_path / 'whole_game_known_good.zip'}, "
            f"not whole_game_latest.zip."
        )
        return False


def tensorboard_dir():
    """TensorBoard if it's installed, plain stdout logging if it isn't.

    Stable-Baselines3 raises on tensorboard_log when the package is missing,
    which would turn "no graphs" into "no training at all". It isn't in this
    project's dependency list (README's install line predates any need for
    it), so it stays optional -- `pip install tensorboard` turns the curves
    on, and nothing breaks without it.
    """
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard not installed -- logging to stdout only "
              "(pip install tensorboard for curves).")
        return None
    return str(TENSORBOARD_DIR)


def make_env(rank):
    def _init():
        env = PokemonRedWholeGameEnv(
            start_state=Path(START_STATE) if START_STATE else None,
            max_steps=MAX_STEPS,
        )
        # Monitor gives SB3 the episode reward/length stats that show up in
        # the logs; without it the run reports nothing useful about how the
        # agent is actually doing.
        return Monitor(env)

    return _init


def latest_checkpoint():
    """Most recent checkpoint, so an interrupted run resumes instead of
    restarting -- the same guarantee initial_state() gives the navigation
    trainer, and it matters more here because the runs are longer.

    Considers whole_game_latest.zip alongside the periodic
    whole_game_<n>_steps.zip files and picks whichever is actually newest by
    mtime -- not the periodic checkpoints alone. latest.zip is written once,
    on exit, and captures the exact step a Ctrl-C landed on; ignoring it here
    would mean resuming from up to CHECKPOINT_EVERY steps before that,
    silently discarding real progress every time a run is stopped and
    restarted.
    """
    if not MODEL_DIR.exists():
        return None

    candidates = list(MODEL_DIR.glob("whole_game_*_steps.zip"))
    latest = MODEL_DIR / "whole_game_latest.zip"
    if latest.exists():
        candidates.append(latest)

    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # spawn, not this platform's fork default. train_navigation_parallel.py
    # already documents why from a real failure: forking after torch has
    # initialized its thread pool parks every worker in a futex wait, and it
    # bites intermittently rather than immediately.
    vec_env = SubprocVecEnv(
        [make_env(i) for i in range(NUM_ENVS)], start_method="spawn"
    )

    resume_from = latest_checkpoint()

    print(f"{NUM_ENVS} parallel emulators, device={DEVICE}, "
          f"{MAX_STEPS} steps per episode, target {TOTAL_STEPS:,} steps.")

    if resume_from is not None:
        print(f"Resuming from {resume_from.name}")
        # POKEMON_RED_LR: override the learning rate for this resume. Added
        # after three of four long continuations from a converged policy
        # collapsed at the training default of 3e-4 -- most damningly the
        # 350M -> 380M run, which changed *nothing* about the reward or
        # observation and still fell from ep_rew_mean ~244 to ~30-78 at
        # ~360M and never recovered (eval delivery 36/36 -> 0/36). A policy
        # that is already good needs small steps, not the exploration-sized
        # ones that got it here. lr_schedule is overridden alongside
        # learning_rate because it is the thing SB3's optimizer actually
        # reads each update -- overriding learning_rate alone leaves the
        # serialized old schedule in charge and changes nothing.
        custom_objects = {}
        lr_override = os.environ.get("POKEMON_RED_LR")
        if lr_override:
            lr = float(lr_override)
            custom_objects["learning_rate"] = lr
            custom_objects["lr_schedule"] = lambda _: lr
        model = PPO.load(
            resume_from, env=vec_env, device=DEVICE,
            custom_objects=custom_objects,
        )
        print(f"Effective learning rate: {model.lr_schedule(1.0)}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            n_steps=N_STEPS,
            batch_size=512,
            n_epochs=3,
            learning_rate=3e-4,
            gamma=0.999,       # long horizon: reward here is thousands of
                               # steps away, unlike the battle envs' ~30
            gae_lambda=0.95,
            ent_coef=0.01,     # keeps exploring rather than committing early
                               # to whatever first earned reward
            clip_range=0.2,
            device=DEVICE,
            tensorboard_log=tensorboard_dir(),
            verbose=1,
        )

    checkpoint_callback = AtomicCheckpointCallback(
        save_freq=max(CHECKPOINT_EVERY // NUM_ENVS, 1),
        save_path=str(MODEL_DIR),
        name_prefix="whole_game",
        verbose=1,
    )
    callbacks = [checkpoint_callback]
    if DELIVERY_GATE_EVERY > 0:
        callbacks.append(DeliveryGateCallback(
            save_path=MODEL_DIR,
            eval_freq=max(DELIVERY_GATE_EVERY // NUM_ENVS, 1),
        ))
    callback = CallbackList(callbacks)

    # SB3's learn() treats total_timesteps as an absolute target only when
    # reset_num_timesteps=True. On resume it instead ADDS the model's own
    # num_timesteps on top (base_class.py's _setup_learn:
    # "total_timesteps += self.num_timesteps") -- so passing TOTAL_STEPS
    # unchanged here would silently retarget a resumed run to
    # TOTAL_STEPS-plus-whatever-it-already-did, past the number printed at
    # startup and past what POKEMON_RED_TOTAL_STEPS documents. Subtracting
    # the resumed count first keeps TOTAL_STEPS meaning what it says on both
    # paths.
    remaining_steps = (
        TOTAL_STEPS if resume_from is None
        else max(TOTAL_STEPS - model.num_timesteps, 0)
    )

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callback,
            reset_num_timesteps=resume_from is None,
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- saving before exit.")
    finally:
        atomic_model_save(model, MODEL_DIR / "whole_game_latest")
        vec_env.close()
        print(f"Saved to {MODEL_DIR / 'whole_game_latest.zip'}")


if __name__ == "__main__":
    # Required on macOS: the default start method here is spawn already, but
    # being explicit keeps this correct if it ever runs on Linux, where fork
    # is the default and is the exact hazard noted above.
    multiprocessing.set_start_method("spawn", force=True)
    main()
