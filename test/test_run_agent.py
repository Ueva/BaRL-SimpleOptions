import pytest

from pytest import approx
from typing import List

from simpleoptions import OptionAgent, BaseOption, BaseEnvironment, PrimitiveOption


class DummyLowerLevelOption(BaseOption):
    def __init__(self, id: int, action: PrimitiveOption):
        super().__init__()
        self.id = id
        self.action = action

    def initiation(self, state):
        return True

    def policy(self, state, test=False):
        return self.action

    def termination(self, state):
        return state in {3, 6}

    def __str__(self):
        return f"LowerLevelDummyOption({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other_option):
        return isinstance(other_option, DummyLowerLevelOption) and self.id == other_option.id

    def __ne__(self, other_option: object):
        return not self == other_option


class DummyHigherLevelOption(BaseOption):
    def __init__(self, id: int, lower_level_option: "DummyLowerLevelOption"):
        super().__init__()
        self.id = id
        self.lower_level_option = lower_level_option

    def initiation(self, state):
        return state in {0, 1, 2, 3, 4, 5, 6}

    def policy(self, state, test=False):
        return self.lower_level_option

    def termination(self, state):
        return state == 6

    def __str__(self):
        return f"HigherLevelDummyOption({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other_option):
        return isinstance(other_option, DummyHigherLevelOption) and self.id == other_option.id

    def __ne__(self, other_option: object):
        return not self == other_option


class DummyEnv(BaseEnvironment):
    def __init__(self):
        super().__init__()

    def reset(self, state=None):
        self.state = 1
        return 1

    def step(self, action, state=None):
        if action == 0:
            self.state = max(1, self.state - 1)
        elif action == 1:
            self.state = min(6, self.state + 1)

        if self.state == 6:
            reward = 1.0
            terminal = True
        else:
            reward = -0.1
            terminal = False

        return self.state, reward, terminal, {}

    def get_action_space(self):
        return [0, 1]

    def get_available_actions(self, state=None):
        return self.get_action_space()

    def is_state_terminal(self, state=None):
        if state is None:
            return self.state == 6
        else:
            return state == 6

    def get_initial_states(self):
        return [0]

    def get_successors(self, state=None):
        if state is None:
            return [max(0, self.state - 1), min(6, self.state + 1)]
        else:
            return [max(0, state - 1), min(6, state + 1)]

    def render(self, mode="human"):
        states = [str(num) for num in range(7)]
        states[self.state] = f"[{self.state}]"
        print(" ".join(states))

    def close(self):
        return


def test_macro_q_update_one_step_higher_level():
    epsilon = 0.0
    macro_alpha = 0.5
    intra_option_alpha = 0.0
    gamma = 0.9
    default_action_value = 0.0
    initial_higher_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    llo1 = DummyLowerLevelOption(1, 1)
    llo2 = DummyLowerLevelOption(2, 1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the higher-level option to 1.0 in the initial
    # state, to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=False,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # Check that the updated value of the higher-level option in the initial state is correct.
    correct_q_value = (1 - macro_alpha) * (initial_higher_level_option_q_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * -0.1 + gamma**3 * -0.1 + gamma**4 * 1.0 + gamma**5 * 0
    )
    assert agent.q_table[(hash(1), hash(hlo1))] == approx(correct_q_value)

    # Check that the value of taking the higher-level option in other states hasn't been changed (from 0).
    assert agent.q_table[(hash(0), hash(hlo1))] == 0
    assert agent.q_table[(hash(2), hash(hlo1))] == 0
    assert agent.q_table[(hash(3), hash(hlo1))] == 0
    assert agent.q_table[(hash(4), hash(hlo1))] == 0
    assert agent.q_table[(hash(5), hash(hlo1))] == 0
    assert agent.q_table[(hash(6), hash(hlo1))] == 0


def test_macro_q_update_n_step_higher_level():
    epsilon = 0.0
    macro_alpha = 0.5
    intra_option_alpha = 0.0
    gamma = 0.9
    default_action_value = 0.0
    initial_higher_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the higher-level option to 1.0 in the initial
    # state, to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=True,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # Check that the updated value of the higher-level option in each state is correct.
    # State 1.
    correct_q_value = (1 - macro_alpha) * (initial_higher_level_option_q_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * -0.1 + gamma**3 * -0.1 + gamma**4 * 1.0 + gamma**5 * 0
    )
    assert agent.q_table[(hash(1), hash(hlo1))] == approx(correct_q_value)

    # State 2.
    correct_q_value = (1 - macro_alpha) * (0) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * -0.1 + gamma**3 * 1.0 + gamma**4 * 0
    )
    assert agent.q_table[(hash(2), hash(hlo1))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - macro_alpha) * (0) + macro_alpha * (-0.1 + gamma * -0.1 + gamma**2 * 1.0 + gamma**3 * 0)
    assert agent.q_table[(hash(3), hash(hlo1))] == approx(correct_q_value)

    # State 4.
    correct_q_value = (1 - macro_alpha) * (0) + macro_alpha * (-0.1 + gamma * 1.0 + gamma**2 * 0)
    assert agent.q_table[(hash(4), hash(hlo1))] == approx(correct_q_value)

    # State 5.
    correct_q_value = (1 - macro_alpha) * (0) + macro_alpha * (1.0 + gamma * 0)
    assert agent.q_table[(hash(5), hash(hlo1))] == approx(correct_q_value)

    # State 0 (unvisited).
    assert agent.q_table[(hash(0), hash(hlo1))] == 0

    # Check that the value of taking the higher-level option in the terminal state hasn't been changed from 0.
    assert agent.q_table[(hash(6), hash(hlo1))] == 0


def test_macro_q_update_one_step_lower_level():
    epsilon = 0.0
    macro_alpha = 0.5
    intra_option_alpha = 0.0
    gamma = 0.9
    default_action_value = 0.0
    initial_higher_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the higher-level option to 1.0 in the initial
    # state, to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=False,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # The lower-level option will terminate in states 3 and 6.
    # So, we should check that its value has been correctly updated in states 1 and 3.
    # State 1.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * default_action_value
    )
    assert agent.q_table[(hash(1), hash(llo1))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * 1.0 + gamma**3 * 0
    )
    assert agent.q_table[(hash(3), hash(llo1))] == approx(correct_q_value)

    # State 0 (unvisited).
    assert agent.q_table[(hash(0), hash(llo1))] == 0

    # Check that the value of taking the lower-level option in other states hasn't been changed (from 0).
    assert agent.q_table[(hash(2), hash(llo1))] == 0
    assert agent.q_table[(hash(4), hash(llo1))] == 0
    assert agent.q_table[(hash(5), hash(llo1))] == 0
    assert agent.q_table[(hash(6), hash(llo1))] == 0

    # Check that the value of taking the other lower-level option hasn't been changed (from 0).
    assert agent.q_table[(hash(0), hash(llo2))] == 0
    assert agent.q_table[(hash(1), hash(llo2))] == 0
    assert agent.q_table[(hash(2), hash(llo2))] == 0
    assert agent.q_table[(hash(3), hash(llo2))] == 0
    assert agent.q_table[(hash(4), hash(llo2))] == 0
    assert agent.q_table[(hash(5), hash(llo2))] == 0
    assert agent.q_table[(hash(6), hash(llo2))] == 0


def test_macro_q_update_n_step_lower_level():
    epsilon = 0.0
    macro_alpha = 0.5
    intra_option_alpha = 0.0
    gamma = 0.9
    default_action_value = 0.0
    initial_higher_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the higher-level option to 1.0 in the initial
    # state, to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=True,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # The lower-level option will terminate in states 3 and 6.
    # Transitions 1 -> 2 -> 3.
    # State 1.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * default_action_value
    )
    assert agent.q_table[(hash(1), hash(llo1))] == approx(correct_q_value)

    # State 2.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (-0.1 + gamma * default_action_value)
    assert agent.q_table[(hash(2), hash(llo1))] == approx(correct_q_value)

    # Transitions 3 -> 4 -> 5 -> 6.
    # State 3.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * 1.0 + gamma**3 * 0
    )
    assert agent.q_table[(hash(3), hash(llo1))] == approx(correct_q_value)

    # State 4.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (-0.1 + gamma * 1.0 + gamma**2 * 0)
    assert agent.q_table[(hash(4), hash(llo1))] == approx(correct_q_value)

    # State 5.
    correct_q_value = (1 - macro_alpha) * (default_action_value) + macro_alpha * (1.0 + gamma * 0)
    assert agent.q_table[(hash(5), hash(llo1))] == approx(correct_q_value)

    # State 0 (unvisited).
    assert agent.q_table[(hash(0), hash(llo1))] == 0

    # Check that the value of taking the lower-level option in the terminal state hasn't been changed from 0.
    assert agent.q_table[(hash(6), hash(llo1))] == 0

    # Check that the value of taking the other lower-level option hasn't been changed (from 0).
    assert agent.q_table[(hash(0), hash(llo2))] == 0
    assert agent.q_table[(hash(1), hash(llo2))] == 0
    assert agent.q_table[(hash(2), hash(llo2))] == 0
    assert agent.q_table[(hash(3), hash(llo2))] == 0
    assert agent.q_table[(hash(4), hash(llo2))] == 0
    assert agent.q_table[(hash(5), hash(llo2))] == 0
    assert agent.q_table[(hash(6), hash(llo2))] == 0


def test_intra_option_update_one_step_higher_level():
    epsilon = 0.0
    macro_alpha = 0.0
    intra_option_alpha = 0.5
    gamma = 0.9
    default_action_value = 0.0
    initial_lower_level_option_q_value = 1.0
    initial_higher_level_option_q_value = 0.1

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the first lower-level option to 1.0 in states 1 and 3
    # to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=False,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(3), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value
    agent.q_table[(hash(3), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # We've executed llo1 in states 1 and 3. hlo1 would also call llo1 in both of these states,
    # so it should have recieved intra-option updates.
    # State 1.
    correct_q_value = (1 - intra_option_alpha) * initial_higher_level_option_q_value + intra_option_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * initial_higher_level_option_q_value
    )
    assert agent.q_table[(hash(1), hash(hlo1))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - intra_option_alpha) * initial_higher_level_option_q_value + intra_option_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * 1.0 + gamma**3 * 0.0
    )
    assert agent.q_table[(hash(3), hash(hlo1))] == approx(correct_q_value)

    # Check that the value of taking the higher-level option in other states hasn't been changed (from 0).
    assert agent.q_table[(hash(0), hash(hlo1))] == 0
    assert agent.q_table[(hash(2), hash(hlo1))] == 0
    assert agent.q_table[(hash(4), hash(hlo1))] == 0
    assert agent.q_table[(hash(5), hash(hlo1))] == 0
    assert agent.q_table[(hash(6), hash(hlo1))] == 0


def test_intra_option_update_n_step_higher_level():
    epsilon = 0.0
    macro_alpha = 0.0
    intra_option_alpha = 0.5
    gamma = 0.9
    default_action_value = 0.0
    initial_lower_level_option_q_value = 1.0
    initial_higher_level_option_q_value = 0.1

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, hlo1])

    # Initialise agent and set the q-value of the first lower-level option to 1.0 in states 1 and 3
    # to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=True,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(3), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(1), hash(hlo1))] = initial_higher_level_option_q_value
    agent.q_table[(hash(3), hash(hlo1))] = initial_higher_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # We've executed llo1 in states 1 and 3. hlo1 would also call llo1 in both of these states
    # and all states in-between, so it should have recieved intra-option updates.
    # State 1.
    correct_q_value = (1 - intra_option_alpha) * initial_higher_level_option_q_value + intra_option_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * initial_higher_level_option_q_value
    )
    assert agent.q_table[(hash(1), hash(hlo1))] == approx(correct_q_value)

    # State 2.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * initial_higher_level_option_q_value
    )
    assert agent.q_table[(hash(2), hash(hlo1))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - intra_option_alpha) * initial_higher_level_option_q_value + intra_option_alpha * (
        -0.1 + gamma * -0.1 + gamma**2 * 1.0 + gamma**3 * 0
    )
    assert agent.q_table[(hash(3), hash(hlo1))] == approx(correct_q_value)

    # State 4.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * 1.0 + gamma**2 * 0
    )
    assert agent.q_table[(hash(4), hash(hlo1))] == approx(correct_q_value)

    # State 5.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (1.0 + gamma * 0)
    assert agent.q_table[(hash(5), hash(hlo1))] == approx(correct_q_value)

    # State 0 (unvisited).
    assert agent.q_table[(hash(0), hash(hlo1))] == 0

    # Check that the value of taking the higher-level option in the terminal state hasn't been changed from 0.
    assert agent.q_table[(hash(6), hash(hlo1))] == 0


def test_intra_option_update_one_step_lower_level():
    epsilon = 0.0
    macro_alpha = 0.0
    intra_option_alpha = 0.5
    gamma = 0.9
    default_action_value = 0.0
    initial_lower_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    llo3 = DummyLowerLevelOption(3, p0)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, llo3, hlo1])

    # Initialise agent and set the q-value of the first lower-level option to 1.0 in states 1 and 3
    # to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=False,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(3), hash(llo1))] = initial_lower_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # llo2 would have taken the same primitive action as llo1 in each state, so
    # it should have recieved intra-option updates.
    # State 1.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(1), hash(llo2))] == approx(correct_q_value)

    # State 2.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * max([agent.q_table[(hash(3), hash(o))] for o in env.get_available_options(3)])
    )
    assert agent.q_table[(hash(2), hash(llo2))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(3), hash(llo2))] == approx(correct_q_value)

    # State 4.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(4), hash(llo2))] == approx(correct_q_value)

    # State 5.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (1.0 + gamma * 0)
    assert agent.q_table[(hash(5), hash(llo2))] == approx(correct_q_value)

    # States 0 and 6.
    assert agent.q_table[(hash(0), hash(llo2))] == 0
    assert agent.q_table[(hash(0), hash(llo2))] == 0

    # llo3 would NOT have taken the same primitive action as llo1 in each state,
    # so it should NOT have recieved intra-option updates.
    assert agent.q_table[(hash(0), hash(llo3))] == 0
    assert agent.q_table[(hash(1), hash(llo3))] == 0
    assert agent.q_table[(hash(2), hash(llo3))] == 0
    assert agent.q_table[(hash(3), hash(llo3))] == 0
    assert agent.q_table[(hash(4), hash(llo3))] == 0
    assert agent.q_table[(hash(5), hash(llo3))] == 0
    assert agent.q_table[(hash(6), hash(llo3))] == 0


def test_intra_option_update_n_step_lower_level():
    epsilon = 0.0
    macro_alpha = 0.0
    intra_option_alpha = 0.5
    gamma = 0.9
    default_action_value = 0.0
    initial_lower_level_option_q_value = 1.0

    # Initialise env and add the dummy options to the list of available options.
    env = DummyEnv()
    p0 = PrimitiveOption(0, env)
    p1 = PrimitiveOption(1, env)
    llo1 = DummyLowerLevelOption(1, p1)
    llo2 = DummyLowerLevelOption(2, p1)
    llo3 = DummyLowerLevelOption(3, p0)
    hlo1 = DummyHigherLevelOption(1, llo1)
    env.set_options([p0, p1, llo1, llo2, llo3, hlo1])

    # Initialise agent and set the q-value of the first lower-level option to 1.0 in states 1 and 3
    # to ensure that it is always chosen (notice that we have disabled exploration).
    agent = OptionAgent(
        env=env,
        epsilon=epsilon,
        macro_alpha=macro_alpha,
        intra_option_alpha=intra_option_alpha,
        gamma=gamma,
        n_step_updates=True,
        default_action_value=default_action_value,
    )
    agent.q_table[(hash(1), hash(llo1))] = initial_lower_level_option_q_value
    agent.q_table[(hash(3), hash(llo1))] = initial_lower_level_option_q_value

    # Run the agent for five time-steps (i.e., until it reaches the terminal state).
    _ = agent.run_agent(num_epochs=1, epoch_length=5)

    # llo2 would have taken the same primitive action as llo1 in each state, so
    # it should have recieved intra-option updates.
    # State 1.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(1), hash(llo2))] == approx(correct_q_value)

    # State 2.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * max([agent.q_table[(hash(3), hash(o))] for o in env.get_available_options(3)])
    )
    assert agent.q_table[(hash(2), hash(llo2))] == approx(correct_q_value)

    # State 3.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(3), hash(llo2))] == approx(correct_q_value)

    # State 4.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (
        -0.1 + gamma * default_action_value
    )
    assert agent.q_table[(hash(4), hash(llo2))] == approx(correct_q_value)

    # State 5.
    correct_q_value = (1 - intra_option_alpha) * (default_action_value) + intra_option_alpha * (1.0 + gamma * 0)
    assert agent.q_table[(hash(5), hash(llo2))] == approx(correct_q_value)

    # States 0 and 6.
    assert agent.q_table[(hash(0), hash(llo2))] == 0
    assert agent.q_table[(hash(0), hash(llo2))] == 0

    # llo3 would NOT have taken the same primitive action as llo1 in each state,
    # so it should NOT have recieved intra-option updates.
    assert agent.q_table[(hash(0), hash(llo3))] == 0
    assert agent.q_table[(hash(1), hash(llo3))] == 0
    assert agent.q_table[(hash(2), hash(llo3))] == 0
    assert agent.q_table[(hash(3), hash(llo3))] == 0
    assert agent.q_table[(hash(4), hash(llo3))] == 0
    assert agent.q_table[(hash(5), hash(llo3))] == 0
    assert agent.q_table[(hash(6), hash(llo3))] == 0
