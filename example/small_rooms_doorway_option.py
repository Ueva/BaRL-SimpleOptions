from simpleoptions import BaseOption
from small_rooms_env import SmallRoomsEnv

# Here, we will manually define a simple option which takes out agent from
# any state in our SmallRooms gridworld environment to the doorway in the middle.


class DoorwayOption(BaseOption):
    def __init__(self):
        return

    def initiation(self, state):
        # Our option should be available in every state except the doorway at (2, 4).
        return state != (2, 4)

    def termination(self, state):
        # Our option should terminate when the agent reaches
        # the doorway grid-cell, at position (2, 4).
        if state == (2, 4):
            return 1.0
        else:
            return 0.0

    def policy(self, state):
        # Here, we'll manually code a simple option policy that takes our agent from its current state to the doorway.
        # In reality, you'd probably learn your option policy using something like Q-Learning or Value Iteration instead of
        # hard-coding it, but this is sufficient for this simple example.

        # If our agent is not on the middle row (row 2), it should move up or down accordingly.
        if state[0] < 2:
            return SmallRoomsEnv.ACTION_IDS["DOWN"]
        elif state[0] > 2:
            return SmallRoomsEnv.ACTION_IDS["UP"]

        # If our agent is not on the middle column (column 4), it should move left or right accordingly.
        if state[1] < 4:
            return SmallRoomsEnv.ACTION_IDS["RIGHT"]
        elif state[1] > 4:
            return SmallRoomsEnv.ACTION_IDS["LEFT"]

    def __str__(self):
        return "DoorwayOption"

    def __repr__(self):
        return "DoorwayOption"

    def __hash__(self):
        return hash(str(self))
