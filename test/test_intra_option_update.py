import pytest

from simpleoptions import (
    OptionAgent,
    BaseOption,
    PrimitiveOption,
    option,
    primitive_option,
)


class DummyOption(BaseOption):
    """
    A dummy option for testing.

    Executes a dummy action (int 1) in every state, terminates with probability 1.0 in a
    given state and probability 0.0 otherwise, and can be initiated in any state.
    """

    def __init__(self, name, action, termination_state):
        self.name = name
        self.action = action
        self.termination_state = termination_state

    def initiation(self, state):
        return True

    def policy(self, state, test=False):
        return self.action

    def termination(self, state):
        if state == self.termination_state:
            return 1.0
        else:
            return 0.0

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(str(self))


class DummyEnv:
    """
    Dummy class to pass in as an environment - implements only what we need.
    """

    def __init__(self, options=[]):
        self.options = options

    def is_state_terminal(self, state):
        return False

    def get_available_actions(self, state):
        return [1]

    def get_available_options(self, state):
        return self.options


# Two options which execute the same primitive action, one time-step, intial q-values of zero.
def test_one_step_intra_option_update_1():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    initial_values = {"state_1, option": 0, "state_2, 1": 0}
    alpha = 0.2
    gamma = 0.9

    # Define two dummy options which execute the SAME lower-level option.
    # If we perform an intra-option update for one of them, the other should get updated.
    lower_level_option = DummyOption("lower_level_option", 1, "state_2")
    option_1 = DummyOption("test_option_1", lower_level_option, "state_2")
    option_2 = DummyOption("test_option_2", lower_level_option, "state_2")

    # Initialise an OptionAgent (note that initial q-values for all states are zero by default).
    agent = OptionAgent(
        env=DummyEnv([option_1, option_2, lower_level_option]),
        macro_alpha=alpha,
        gamma=gamma,
    )

    # First, perform a macro-q update for option_1. Then, perform an intra-option update for
    # option_1. This should result in both option_1 and option_2 having the same q-value in
    # state_1, since they both execute the same primitive action, and should both get updated once.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option_1, n_step=True)
    agent.intra_option_learn(state_trajectory, reward_trajectory, lower_level_option, option_1, n_step=True)

    # Ensure that the q-values for executing option_1 in state_1 and executing option_2 in state_1 are the same.
    assert agent.q_table[(hash("state_1"), hash(option_1))] == agent.q_table[(hash("state_1"), hash(option_2))]


# Two options which execute the different primitive actions, one time-step, initial q-values of zero.
def test_one_step_intra_option_update_2():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    initial_values = {"state_1, option": 0, "state_2, 1": 0}
    alpha = 0.2
    gamma = 0.9

    # Define two dummy options which execute DIFFERENT options. If we
    # perform an intra-option update for one of them, the other should NOT get updated.
    lower_level_option_1 = DummyOption("lower_level_option_1", 1, "state_2")
    option_1 = DummyOption("test_option_1", lower_level_option_1, "state_2")
    lower_level_option_2 = DummyOption("lower_level_option_2", 2, "state_2")
    option_2 = DummyOption("test_option_2", lower_level_option_2, "state_2")

    # Initialise an OptionAgent (note that initial q-values for all states are zero by default).
    agent = OptionAgent(
        env=DummyEnv([option_1, option_2, lower_level_option_1, lower_level_option_2]),
        macro_alpha=alpha,
        gamma=gamma,
    )

    # First, perform a macro-q update for option_1. Then, perform an intra-option update for
    # option_1. This should result in option_1 having its q-values updated once (during the
    # macro-q update), but option_2 should not be updated during the intra-option update because
    # its policy is different to option_1's.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option_1, n_step=True)
    agent.intra_option_learn(state_trajectory, reward_trajectory, lower_level_option_1, option_1, n_step=True)

    # Ensure that the q-value of executing option_1 in state_1 is updated, and that the
    # q-value of executing option_2 in state_1 remains at 0.0.
    assert agent.q_table.get((hash("state_1"), hash(option_1)), 0) == 0.2
    assert agent.q_table.get((hash("state_1"), hash(option_2)), 0) == 0.0
