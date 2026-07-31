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

from envs.battle_env import PokemonRedRivalBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "rival_battle_dqn.zip"


def main(total_timesteps=20000):
    env = Monitor(PokemonRedRivalBattleEnv(max_steps=30))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=20000,
        learning_starts=200,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        target_update_interval=250,
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
