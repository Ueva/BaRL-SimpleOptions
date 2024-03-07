from simpleoptions import BaseOption


class PrimitiveOption(BaseOption):
    def __init__(self, action):
        self.action = action

    def initiation(self, state):
        return True

    def policy(self, state, test=False):
        return self.action

    def termination(self, state):
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
