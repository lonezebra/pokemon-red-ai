from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from envs.wild_battle_env import PokemonRedWildBattleEnv
from core.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "wild_battle_dqn.zip"


def main(total_timesteps=20000):
    # Same DQN setup as train_battle_agent.py (rival battle) -- same
    # observation shape (6 floats) and only one more discrete action
    # (run), so there's no reason to expect this needs different
    # hyperparameters until evaluation says otherwise.
    env = Monitor(PokemonRedWildBattleEnv(max_steps=30))

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
