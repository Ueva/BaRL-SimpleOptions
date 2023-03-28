from typing import Hashable

from simpleoptions.option import Option
from simpleoptions.environment import BaseEnvironment


class PrimitiveOption(Option):
    """
    Class representing a primitive option.
    Primitive options terminate with probability one in every state, and have
    an initiation set consisting of all of the states where their underlying
    primitive actions are available.
    """

    def __init__(self, action: Hashable, env: "BaseEnvironment"):
        """Constructs a new primitive option.

        Arguments:
            action {Hashable} -- The underlying primitive action for this option.
            env {BaseEnvironment} -- The environment that this environment is to be executed in.
        """
        self.env = env
        self.action = action

    def initiation(self, state: Hashable) -> bool:
        return self.action in self.env.get_available_actions(state)

    def policy(self, state: Hashable) -> Hashable:
        return self.action

    def termination(self, state: Hashable) -> float:
        return True

    def __str__(self):
        return f"PrimitiveOption({hash(self.action)})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other_option):
        if isinstance(other_option, PrimitiveOption):
            return other_option.action == self.action
        else:
            return False

    def __ne__(self, other_option):
        return not self == other_option
