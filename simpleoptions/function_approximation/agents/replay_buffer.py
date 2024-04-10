from random import choices
from collections import deque, namedtuple

PrimitiveTransition = namedtuple("Transition", ("state", "action", "reward", "next_state", "terminal"))


class ReplayBuffer(object):
    def __init__(self, max_size, experience=None):
        if experience is None:
            self.buffer = deque(maxlen=max_size)
        else:
            self.buffer = deque(experience, maxlen=max_size)

        self.max_size = max_size

    def add(self, state, action, reward, next_state, terminal):
        self.buffer.append((state, action, reward, next_state, terminal))

    def sample(self, batch_size=1):
        return list(choices(self.buffer, k=batch_size))

    def __len__(self):
        return len(self.buffer)
