import numpy as np

from copy import deepcopy

from simpleoptions import BaseEnvironment

#  The environment's layout, consisting of two 3x3 rooms connected by a small doorway.
#  The start is in the top-left, the goal is in the bottom-right.
#  Rewards of -1 per time-step, +10 for reaching the goal.
#  Deterministic actions for moving North, South, East, and West.
#  Moving into a wall causes the agent to stay in its current state, but time advances.
#  # # # # # # # # #
#  # S . . # . . . #
#  # . . . . . . . #
#  # . . . # . . G #
#  # # # # # # # # #
#
#  . = Passable Floor, # = Impassable Wall, S = start, G = Goal


class SmallRoomsEnv(BaseEnvironment):
    # Mapping from action IDs to human-readable descriptions.
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    ACTION_IDS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}

    def __init__(self, options=[]):
        super().__init__(options)

        # Initialise an array representing the gridworld.
        self.rooms = self._initialise_rooms()

        # Define the coordinates of the start and goal states.
        self.start_state = (1, 1)
        self.goal_state = (3, 7)
        self.current_state = None

    def reset(self):
        # Reset the agent to be at the current state.
        self.current_state = self.start_state
        return self.start_state

    def step(self, action):
        # Compute which grid-cell the agent's intended action takes it to.
        self.current_state = self._get_intended_cell(self.current_state, action)

        # Compute the agent's reward (-1, plus 10 if it has reached the goal).
        reward = -1
        if self.current_state == self.goal_state:
            reward += 10

        # Return (next_state, reward, terminal, info).
        return (
            self.current_state,
            reward,
            self.is_state_terminal(self.current_state),
            {},
        )

    def get_action_space(self):
        # The agent has four actions (up, down, left, right).
        return list([0, 1, 2, 3])

    def get_available_actions(self, state):
        # The agent has access to all four actions in every state.
        return self.get_action_space()

    def is_state_terminal(self, state):
        # The state is only terminal if the agent has reached the goal.
        return self.current_state == self.goal_state

    def get_initial_states(self):
        # The agent can only start at the state state.
        return deepcopy([self.start_state])

    def get_successors(self, state=None, actions=None):
        if state is None:
            state = self.current_state
        if actions is None:
            actions = self.get_available_actions(state)

        successors = []
        for action in actions:
            # Compute which grid-cell the agent's intended action takes it to.
            sucessor_state = self._get_intended_cell(state, action)

            # Add unique successor states to the list of possible successor states.
            if sucessor_state not in successors:
                successors.append(sucessor_state)

        return successors

    def render(self, mode="human"):
        # Prints a simple representation of the environment's current state to the console.
        self.rooms[self.current_state] = "A"
        for i in range(self.rooms.shape[0]):
            print("".join(self.rooms[i]))
        print()
        self.rooms[self.current_state] = "."

    def close(self):
        pass

    def _initialise_rooms(self):
        # Builds an array representing the two-room gridworld.
        rooms = np.full((5, 9), "#", dtype=str)
        rooms[1:-1, 1:-1] = "."
        rooms[:, 4] = "#"
        rooms[2, 4] = "."
        return rooms

    def _get_intended_cell(self, current_state, action):
        intended_next_state = current_state

        # Change the agent's next position based on the direction it wants to move in.
        if self.ACTION_NAMES[action] == "UP":
            intended_next_state = (intended_next_state[0] - 1, intended_next_state[1])
        elif self.ACTION_NAMES[action] == "DOWN":
            intended_next_state = (intended_next_state[0] + 1, intended_next_state[1])
        elif self.ACTION_NAMES[action] == "LEFT":
            intended_next_state = (intended_next_state[0], intended_next_state[1] - 1)
        elif self.ACTION_NAMES[action] == "RIGHT":
            intended_next_state = (intended_next_state[0], intended_next_state[1] + 1)

        # If the agent has moved into a wall, return it to where it tried to move from.
        if self.rooms[intended_next_state] == "#":
            intended_next_state = current_state

        return intended_next_state
