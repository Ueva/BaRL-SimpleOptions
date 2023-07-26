import random
import copy

from simpleoptions import BaseEnvironment, BaseOption
from simpleoptions.implementations import GenericOptionGenerator

from typing import List, Hashable, Set, Dict
from abc import abstractmethod

from collections import defaultdict
from itertools import cycle


class SubgoalOptionGenerator(GenericOptionGenerator):
    def __init__(
        self,
        option_learning_alpha: float,
        option_learning_epsilon: float,
        option_learning_gamma: float,
        option_learning_max_steps: int,
        option_learning_max_episode_steps: int,
        option_learning_default_action_value: float,
    ):
        self.alpha = option_learning_alpha
        self.epsilon = option_learning_epsilon
        self.gamma = option_learning_gamma
        self.max_steps = option_learning_max_steps
        self.max_episode_steps = option_learning_max_episode_steps
        self.default_action_value = option_learning_default_action_value

    @abstractmethod
    def generate_options(self, env: BaseEnvironment, directed: bool, goal_states: List[Hashable] = None):
        pass

    def train_option(self, option: "SubgoalOption"):
        q_table = defaultdict(lambda: self.default_action_value)

        # Create a list of states in the initation set to draw from.
        initial_states = list(option.initiation_set) * 5
        random.shuffle(initial_states)
        initial_states = cycle(initial_states)

        time_steps = 0
        while time_steps < self.max_steps:
            # Choose (non-terminal!) state in the initiation set.
            state = next(
                option.env.reset(state)
                for state in initial_states
                if not option.env.is_state_terminal(state) and state != option.subgoal
            )
            episode_steps = 0
            terminal = False

            while not terminal:
                # Select and execute action.
                action = self._select_action(state, option, q_table)
                next_state, _, done, _ = option.env.step(action.policy(state))
                time_steps += 1
                episode_steps += 1

                # Compute reward and terminality.
                if next_state == option.subgoal:  # Agent reached subgoal.
                    reward = 1.0
                    terminal = True
                elif next_state not in option.initiation_set:  # Agent left the initiation set.
                    reward = -1.0
                    terminal = True
                elif done:  # Agent reached a terminal state.
                    reward = -1.0
                    terminal = True
                else:  # Otherwise...
                    reward = -0.001
                    terminal = False

                # Perform Q-Learning update.
                old_q = q_table[(hash(state), hash(action))]
                max_next_q = (
                    0
                    if terminal
                    else max(
                        [
                            q_table[(hash(next_state), hash(next_action))]
                            for next_action in option.env.get_available_options(next_state)
                        ]
                    )
                )
                new_q = reward + self.gamma * max_next_q
                q_table[(hash(state), hash(action))] = old_q + self.alpha * (new_q - old_q)

                state = next_state

                if (
                    episode_steps > self.max_episode_steps or time_steps > self.max_steps
                ):  # Training time-limit exceeded.
                    break

        option.q_table = q_table

    def _select_action(self, state: Hashable, option: "SubgoalOption", q_table: Dict):
        available_actions = [action for action in option.env.get_available_options(state)]

        # Exploratory action.
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        # Greedy action, tie-breaking randomly.
        else:
            max_value = max([q_table[(hash(state), hash(action))] for action in available_actions])
            best_actions = [action for action in available_actions if q_table[(hash(state), hash(action))] == max_value]
            return random.choice(best_actions)


class SubgoalOption(BaseOption):
    def __init__(self, env: BaseEnvironment, subgoal: Hashable, initiation_set: Set[Hashable], q_table: Dict = None):
        self.env = copy.copy(env)
        self.subgoal = subgoal
        self.initiation_set = initiation_set

        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table

    def initiation(self, state):
        return state in self.initiation_set

    def termination(self, state):
        return state == self.subgoal or state not in self.initiation_set

    def policy(self, state, test=False):
        # Return highest-valued option from the Q-table, breaking ties randomly.
        available_actions = self.env.get_available_options(state)
        max_value = max([self.q_table.get((hash(state), hash(action)), 0) for action in available_actions])
        return random.choice(
            [option for option in available_actions if self.q_table[(hash(state), hash(option))] == max_value]
        )

    def __str__(self):
        return f"SubgoalOption({self.subgoal})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, SubgoalOption):
            return self.subgoal == other.subgoal
        else:
            return False

    def __ne__(self, other):
        return not self == other
