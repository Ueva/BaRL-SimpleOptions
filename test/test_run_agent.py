import pytest

from typing import List

from simpleoptions import OptionAgent, BaseOption, BaseEnvironment


class DummyLowerLevelOption(BaseOption):
    def __init__(self, id: int):
        super().__init__()
        self.id = id

    def initiation(self, state):
        return True

    def policy(self, state, test=False):
        return 1

    def termination(self, state):
        return state in {3, 6}

    def __str__(self):
        return f"LowerLevelDummyOption({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other_option):
        return isinstance(other_option, DummyLowerLevelOption)

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

