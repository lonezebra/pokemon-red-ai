import os

# Headless unless explicitly asked otherwise, and this must run before any
# core import, because core/config.py reads the variable at import time.
#
# The default is "SDL2", which opens a real Game Boy window. That is what
# you want for watch_*.py, and actively harmful here: it renders every
# frame to the screen at EMULATION_SPEED=0, i.e. as fast as the emulator
# can produce them, dragging a training run that should take minutes into
# a visibly "hung" one, and popping up an unwanted window on a machine
# this might be run on unattended. GIFs are unaffected: the screen buffer
# is still readable with no window.
os.environ.setdefault("POKEMON_AI_WINDOW_MODE", "null")

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from envs.wild_battle_env import PokemonRedWildBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "wild_battle_dqn.zip"


def main(total_timesteps=200000):
    # Same DQN setup as train_battle_agent.py (rival battle), still --
    # only the timestep/buffer budget changed. 50000 steps produced a
    # model that never once attempted to catch anything in 100 greedy
    # eval episodes: even at low enemy HP (the best real catch chance),
    # its own Q-values rated CATCH at ~2.3 against ~8.9 for fighting --
    # not a narrow miss, a policy that never learned catching has any
    # real value. The likely cause isn't the reward shape (that part
    # was verified correct in isolation) but exploration: catching only
    # ever pays off after a multi-step sequence -- fight down HP, *then*
    # switch to catch -- and plain epsilon-greedy stumbles onto that
    # exact compound sequence far less often than it stumbles onto
    # "just fight" or "just run", which pay off after a single action.
    # 50000 steps (~1600 episodes) wasn't enough experience of that
    # sequence for the Q-function to learn its true, HP-dependent
    # value. Raised 4x, with the replay buffer raised alongside it so
    # the rarer catch-sequence transitions it does see stick around for
    # more replay passes instead of being evicted quickly.
    #
    # learning_rate and target_update_interval changed again after that
    # 200k run: mixing weakened-HP reset states into the pool (see
    # wild_battle_env.py) produces far more short, high-reward episodes,
    # and at 1e-3/250 the result was outright divergence, not a slow
    # learner -- eval showed Q-values around 25-30 against a reward
    # scale that tops out near 15-17, and the greedy policy had
    # collapsed onto spamming Tail Whip (a zero-damage status move)
    # every single turn regardless of state, going 0-for-100. SB3's
    # core DQN bootstraps off its own target network's max (it isn't
    # Double DQN), so it always carries some overestimation bias; a
    # high learning rate plus frequent target syncs let that bias
    # compound faster than it could correct, and the new pool's glut of
    # short high-value episodes gave it more to run away on. Lowering
    # the learning rate 10x and syncing the target net 4x less often
    # verified clean at both 4k and 15k steps (sane Q-value magnitudes,
    # varying argmax across states, no collapse) before committing to
    # spending another full-length run on it.
    env = Monitor(PokemonRedWildBattleEnv(max_steps=30))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=200,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(str(MODEL_PATH))
    print(f"Saved model to {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    main()
