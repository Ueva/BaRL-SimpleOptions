from typing import Hashable

from simpleoptions.option import BaseOption
from simpleoptions.environment import BaseEnvironment


class PrimitiveOption(BaseOption):
    """
    Class representing a primitive option.
    Primitive options terminate with probability one in every state, and have
    an initiation set consisting of all of the states where their underlying
    primitive actions are available.
    """

    # work around that fixes issues with Dill load (specifically overloading hash) \
    # https://stackoverflow.com/questions/75409930/pickle-and-dill-cant-load-objects-with-overridden-hash-function-attributee
    action = ""

    def __init__(self, action: Hashable, env: "BaseEnvironment"):
        """Constructs a new primitive option.

        Arguments:
            action {Hashable} -- The underlying primitive action for this option.
            env {BaseEnvironment} -- The environment that this environment is to be executed in.
        """
        self.env = env
        self.action = action

        # Constructs the initiation set for this primitive option.
        self.initiation_set = set()
        for state in self.env.get_state_space():
            if self.action in self.env.get_available_actions(state=state):
                self.initiation_set.add(state)

    def initiation(self, state: Hashable) -> bool:
        return self.action in self.initiation_set

    def policy(self, state: Hashable, test: bool = False) -> Hashable:
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
