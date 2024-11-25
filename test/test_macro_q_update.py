from simpleoptions import BaseOption, OptionAgent


class DummyOption(BaseOption):
    """
    A dummy option for testing.

    Returns a dummy action, (int 1), terminates with probability 1.0 in
    every state, and can be initiated in any state.
    """

    def __init__(self, name):
        self.name = name

    def initiation(self, state):
        return True

    def policy(self, state, test=False):
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


# Test a macro_q_learning update over a single time-step, with initial q-values of zero.
def test_one_step_macro_q_update_1():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    option = DummyOption("test_option_1")
    initial_values = {"state_1, option": 0, "state_2, 1": 0}
    alpha = 0.2
    gamma = 0.9

    # Initialise an OptionAgent (note that initial q-values for all states are zero by default).
    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)

    # Perform a macro-q learning update for our test option over a single time-step, where a reward of
    # 1 was earned after transitioning from state_1 to state_2.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option, n_step=True)

    # Ensure that the new value of initating our test option in state_1 is correct after performing the update.
    # Our dummy environment returns a list containing the primitive action 1 as the available action in state_2,
    # and its value will be zero (we assume that this is the first update, and initial q-valaues are zero by default).
    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0] + gamma * initial_values["state_2, 1"] - initial_values["state_1, option"]
    )


# Test a macro_q_learning update over a single time-step, with non-zero initial q-values.
def test_one_step_macro_q_update_2():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2"]
    reward_trajectory = [1]
    option = DummyOption("test_option_1")
    initial_values = {"state_1, option": 2, "state_2, 1": 4}
    alpha = 0.2
    gamma = 0.9

    # Initialise an OptionAgent..
    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)

    # This time, our initial q_values are not zero. We set them to their test values here.
    # Our dummy environment returns a list containing the primitive action 1 as the available action
    # in state_2. This action's value is set to be greater than zero, so it will be the best-valued action.
    agent.q_table[(hash("state_1"), hash(option))] = initial_values["state_1, option"]
    agent.q_table[hash("state_2"), hash(1)] = initial_values["state_2, 1"]

    # Perform a macro-q learning update for our test option over a single time-step, where a reward of
    # 1 was earned after transitioning from state_1 to state_2.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option, n_step=True)

    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0] + gamma * initial_values["state_2, 1"] - initial_values["state_1, option"]
    )


# Test a macro_q_learning update over three time-steps, with initial q-values of zero.
def test_three_step_macro_q_update_1():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2", "state_3", "state_4"]
    reward_trajectory = [2, 3, 4]
    option = DummyOption("test_option_1")
    initial_values = {
        "state_1, option": 0,
        "state_2, option": 0,
        "state_3, option": 0,
        "state_4, 1": 0,
    }
    alpha = 0.2
    gamma = 0.9

    # Initialise an OptionAgent..
    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)

    # Perform a macro-q learning update for our test option over three time-steps, as the option
    # executes and causes our agent to navigate between the states in state_trajectory and earn
    # the rewards in reward_trajectory.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option, n_step=True)

    # Ensure that the value of executing our test option in state_1 has been updated correctly.
    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0]
        + gamma**1 * reward_trajectory[1]
        + gamma**2 * reward_trajectory[2]
        + gamma**3 * initial_values["state_4, 1"]
        - initial_values["state_1, option"]
    )

    # Ensure that the value of executing our test option in state_2 has been updated correctly.
    assert agent.q_table[(hash("state_2"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[1]
        + gamma**1 * reward_trajectory[2]
        + gamma**2 * initial_values["state_4, 1"]
        - initial_values["state_2, option"]
    )

    # Ensure that the value of executing our test option in state_3 has been updated correctly.
    assert agent.q_table[(hash("state_3"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[2] + gamma**1 * initial_values["state_4, 1"] - initial_values["state_3, option"]
    )


# Test a macro_q_learning update over three time-steps, with non-zero initial q-values.
def test_three_step_macro_q_update_2():
    # Set up dummy states, rewards, options, parameters etc. for the test.
    state_trajectory = ["state_1", "state_2", "state_3", "state_4"]
    reward_trajectory = [2, 3, 4]
    option = DummyOption("test_option_1")
    initial_values = {
        "state_1, option": 4,
        "state_2, option": 5,
        "state_3, option": 6,
        "state_4, 1": 7,
    }
    alpha = 0.2
    gamma = 0.9

    agent = OptionAgent(DummyEnv(), macro_alpha=alpha, gamma=gamma)

    # This time, our initial q_values are not zero. We set them to their test values here.
    # Our dummy environment returns a list containing the primitive action 1 as the available action
    # in state_4. This action's value is set to be greater than zero, so it will be the best-valued action.
    agent.q_table[(hash("state_1"), hash(option))] = initial_values["state_1, option"]
    agent.q_table[(hash("state_2"), hash(option))] = initial_values["state_2, option"]
    agent.q_table[(hash("state_3"), hash(option))] = initial_values["state_3, option"]
    agent.q_table[hash("state_4"), hash(1)] = initial_values["state_4, 1"]

    # Perform a macro-q learning update for our test option over three time-steps, as the option
    # executes and causes our agent to navigate between the states in state_trajectory and earn
    # the rewards in reward_trajectory.
    agent.macro_q_learn(state_trajectory, reward_trajectory, option, n_step=True)

    # Ensure that the value of executing our test option in state_1 has been updated correctly.
    assert agent.q_table[(hash("state_1"), hash(option))] == initial_values["state_1, option"] + alpha * (
        reward_trajectory[0]
        + gamma**1 * reward_trajectory[1]
        + gamma**2 * reward_trajectory[2]
        + gamma**3 * initial_values["state_4, 1"]
        - initial_values["state_1, option"]
    )

    # Ensure that the value of executing our test option in state_2 has been updated correctly.
    assert agent.q_table[(hash("state_2"), hash(option))] == initial_values["state_2, option"] + alpha * (
        reward_trajectory[1]
        + gamma**1 * reward_trajectory[2]
        + gamma**2 * initial_values["state_4, 1"]
        - initial_values["state_2, option"]
    )

    # Ensure that the value of executing our test option in state_3 has been updated correctly.
    assert agent.q_table[(hash("state_3"), hash(option))] == initial_values["state_3, option"] + alpha * (
        reward_trajectory[2] + gamma**1 * initial_values["state_4, 1"] - initial_values["state_3, option"]
    )
