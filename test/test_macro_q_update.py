import pytest

from copy import deepcopy

from barl_simpleoptions.option import Option
from barl_simpleoptions.options_agent import OptionAgent


class DummyOption(Option):
    """
    A dummy option for testing.

    Returns a dummy action, (int 1), terminates with probability 1.0 in
    every state, and can be initiated in any state.
    """

    def __init__(self, name):
        self.name = name

    def initiation(self, state):
        return True

    def policy(self, state):
        return 1

    def termination(self, state):
        return 1.0

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))


class DummyEnv:
    """
    Dummy class to pass in as an environment - implements only what we need.
    """

    def __init__(self):
        return

    def is_state_terminal(self, state):
        return False

    def get_available_options(self, state):
        return [1]


# Test a macro_q_learning update over a single time-step.
def test_one_step_macro_q_update_1():
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    option = DummyOption("test_option_1")
    initial_values = {"state_1, option": 0, "state_2, option": 0}
    alpha = 0.2
    gamma = 0.9

    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)
    agent.macro_q_learn(state_trajectory, reward_trajectory, option)
    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0] + gamma * initial_values["state_2, option"] - initial_values["state_1, option"]
    )


# Test a macro_q_learning update over a single time-step, with non-zero initial q-values.
def test_one_step_macro_q_update_2():
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    option = DummyOption("test_option_1")
    initial_values = {"state_1, option": 2, "state_2, option": 4}
    alpha = 0.2
    gamma = 0.9

    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)
    agent.q_table[(hash("state_1"), hash(option))] = initial_values["state_1, option"]
    agent.q_table[(hash("state_2"), hash(option))] = initial_values["state_2, option"]
    agent.q_table[hash("state_2"), hash(1)] = initial_values["state_2, option"]
    agent.macro_q_learn(state_trajectory, reward_trajectory, option)
    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0] + gamma * initial_values["state_2, option"] - initial_values["state_1, option"]
    )