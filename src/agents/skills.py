"""
Uniform wrappers around each trained skill.

The whole point of these classes is that a controller chaining skills
together shouldn't need to know or care whether a given skill is backed
by a lookup table (the leave-house Q-agent) or a neural network (the
rival-battle DQN) -- it should just be able to call
`skill.choose_action(observation)` and get back an action, the same way,
every time.
"""

from stable_baselines3 import DQN

from agents.q_learning_agent import QLearningAgent
from actions import num_actions


class LeaveHouseSkill:
    """Wraps the trained leave-house Q-agent."""

    def __init__(self, q_table_path):
        self.agent = QLearningAgent(num_actions=num_actions())
        self.agent.load(q_table_path)

    def choose_action(self, observation):
        # greedy=True: use what's been learned, no random exploration --
        # exploration is a training-time concern, not a deployed one.
        return self.agent.choose_action(observation, greedy=True)


class RivalBattleSkill:
    """Wraps the trained rival-battle DQN."""

    def __init__(self, model_path):
        self.model = DQN.load(str(model_path))

    def choose_action(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)
