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


class QTableSkill:
    """
    Wraps any trained tabular Q-agent -- leave-house, Route 1/2/3, the
    forest -- all of them are the same (map_id, x, y)-keyed lookup table
    under QLearningAgent, differing only in which JSON file was trained.
    Named for what it wraps rather than any one skill (it used to be
    LeaveHouseSkill, back when that was the only tabular skill this
    controller chained) now that the controller reuses it for every
    navigation segment.
    """

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
