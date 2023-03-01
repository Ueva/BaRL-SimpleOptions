import gc
import math
import random
import numpy as np

from copy import copy, deepcopy
from collections import defaultdict
from typing import Hashable, List, Union

from barl_simpleoptions.option import Option
from barl_simpleoptions.environment import BaseEnvironment


class OptionAgent:
    """
    An agent which acts in a given environment, learning using the Macro-Q learning
    and intra-option learning algorithms.
    """

    def __init__(
        self,
        env: "BaseEnvironment",
        epsilon: float = 0.15,
        macro_alpha: float = 0.2,
        intra_option_alpha: float = 0.2,
        gamma: float = 1.0,
        default_action_value=0.0,
        n_step_updates=False,
    ):
        """
        Constructs a new OptionAgent object.

        Arguments:
            env {Environment} -- The environment for the agent to act in.
            epsilon {float} -- The chance of the agent taking a random action when following its base policy. Defaults to 0.15.
            alpha {float} -- The learning rate used in the Macro-Q Learning updates. Defaults to 0.2.
            alpha {float} -- The learning rate used in the Intra-Option Learning updates. Deafults to 0.2.
            gamma {float} -- The environment's discount factor. Defaults to 1.0.
            default_action_value {float} -- The value to initialise all action-values to. Defaults to 0.0.
            n_step_updates {bool} -- Whether to perform n-step updates (not guaranteed to be consistent at this time). Defaults to False.
        """

        self.q_table = defaultdict(lambda: default_action_value)
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.macro_q_alpha = macro_alpha
        self.intra_option_alpha = intra_option_alpha
        self.executing_options = []
        self.executing_options_states = []
        self.executing_options_rewards = []
        self.n_step_updates = n_step_updates

    def macro_q_learn(
        self, state_trajectory: List[Hashable], rewards: List[float], option: "Option", n_step=False
    ) -> None:
        """
        Performs Macro Q-Learning updates along the given trajectory for the given Option.

        Args:
            state_trajectory (List[Hashable]): The list of states visited each time-step while the option was executing.
            rewards (List[float]): The list of rewards earned each time-step while the Option was executing.
            option (Option): The option to perform an update for.
            n_step (bool): Whether or not to perform n-step updates. Defaults to False, performing one-step updates.
        """
        state_trajectory = deepcopy(state_trajectory)
        rewards = deepcopy(rewards)
        option = option

        # For Debuging - saves info about long-running options to a file.
        if (len(state_trajectory) > 500) and hasattr(option, "hierarchy_level"):
            with open("long_running_options.txt", "a+") as f:
                # Write the ID of the option and how long it executed for.
                f.write(
                    f"Option {option.hierarchy_level}-{option.source_cluster}->{option.target_cluster} ran for {len(state_trajectory)} decision stages.\n"
                )
                # Write the q-values in the state it got stuck in.
                most_common_state = max(set(state_trajectory), key=state_trajectory.count)
                q_values = {
                    str(o): option.q_table.get((hash(most_common_state), hash(o)), 0)
                    for o in self.env.get_available_options(most_common_state)
                }
                f.write(f"State Stuck: {most_common_state}\tQ-Values: {q_values}\n\n")
                quit()

        termination_state = state_trajectory[-1]

        while len(state_trajectory) > 1:
            num_rewards = len(rewards)
            initiation_state = state_trajectory[0]

            old_value = self.q_table[(hash(initiation_state), hash(option))]

            # Compute discounted sum of rewards.
            discounted_sum_of_rewards = self._discounted_return(rewards, self.gamma)

            # Get Q-Values for Next State.
            if not self.env.is_state_terminal(termination_state):
                q_values = [
                    self.q_table[(hash(termination_state), hash(o))]
                    for o in self.env.get_available_options(termination_state)
                ]
            # Cater for terminal states (Q-value is zero).
            else:
                q_values = [0]

            # Perform Macro-Q Update
            self.q_table[(hash(initiation_state), hash(option))] = old_value + self.macro_q_alpha * (
                discounted_sum_of_rewards + math.pow(self.gamma, len(rewards)) * max(q_values) - old_value
            )

            state_trajectory.pop(0)
            rewards.pop(0)

            # If we're not performing n-step updates, exit after the first iteration.
            if not n_step:
                break

    def intra_option_learn(
        self,
        state_trajectory: List[Hashable],
        rewards: List[float],
        executed_option: Option,
        higher_level_option: Union["Option", None] = None,
        n_step=False,
    ) -> None:
        """
        Performs Intra-Option Learning updates along the given trajectory for the given Option.

        Args:
            state_trajectory (List[Hashable]): The list of states visited each time-step while the option was executing.
            rewards (List[float]): The list of rewards earned each time-step while the option was executing.
            executed_option (Option): The option that was executed.
            higher_level_option (Union[None, optional): The option whose policy chose the executed_option. Defaults to None, indicating that the option was executed under the base policy.
            n_step (bool): Whether or not to perform n-step updates. Defaults to False, performing one-step updates.
        """
        state_trajectory = deepcopy(state_trajectory)
        rewards = deepcopy(rewards)
        executed_option = executed_option

        termination_state = state_trajectory[-1]

        while len(state_trajectory) > 1:
            num_rewards = len(rewards)
            initiation_state = state_trajectory[0]

            # We perform an intra-option update for all other options which select executed_option in this state.
            for other_option in self.env.get_available_options(initiation_state):
                if (
                    (hash(other_option) != hash(higher_level_option) or higher_level_option is None)
                    and other_option.initiation(initiation_state)
                    and hash(other_option.policy(initiation_state)) == hash(executed_option)
                ):
                    old_value = self.q_table[(hash(initiation_state), hash(other_option))]

                    # Compute discounted sum of rewards.
                    discounted_sum_of_rewards = self._discounted_return(rewards, self.gamma)

                    if not self.env.is_state_terminal(termination_state):
                        # If the option terminates, we consider the value of the next best option.
                        next_q_terminates = other_option.termination(termination_state) * max(
                            [
                                self.q_table[(hash(termination_state), hash(o))]
                                for o in self.env.get_available_options(termination_state)
                            ]
                        )
                        # If the option continues, we consider the value of the currently executing option.
                        next_q_continues = (1 - other_option.termination(termination_state)) * self.q_table[
                            (hash(termination_state), hash(other_option))
                        ]

                    else:
                        next_q_terminates = 0
                        next_q_continues = 0

                    # Perform Intra-Option Update.
                    self.q_table[(hash(initiation_state), hash(other_option))] = old_value + self.intra_option_alpha * (
                        discounted_sum_of_rewards
                        + math.pow(self.gamma, len(rewards)) * (next_q_continues + next_q_terminates)
                        - old_value
                    )

            state_trajectory.pop(0)
            rewards.pop(0)

            # If we're not performing n-step updates, exit after the first iteration.
            if not n_step:
                break

    def select_action(self, state: Hashable) -> Union[Option, Hashable, None]:
        """
        Returns the selected option for the given state.

        Arguments:
            state {Hashable} -- The state in which to select an option.

        Returns:
            {Option, Hashable, None} -- Returns an Option, or a Primitive Action (as a Hashable).
        """

        # Select option from set of available options
        # Use epsilon greedy at lowest level, use option policy at higher levels.

        # If we do not currently have any options executing, we act according to the agent's
        # base epsilon-greedy policy over the set of currently available options.
        if len(self.executing_options) == 0:
            # Random Action.
            if random.random() < self.epsilon:
                available_options = self.env.get_available_options(state, exploration=True)
                return random.choice(available_options)
            # Best Action.
            else:
                available_options = self.env.get_available_options(state, exploration=False)
                # Find Q-values of available options.
                q_values = [self.q_table[(hash(state), hash(o))] for o in available_options]

                # Return the option with the highest Q-value, breaking ties randomly.
                return available_options[
                    random.choice([idx for idx, q_value in enumerate(q_values) if q_value == max(q_values)])
                ]
        # If we are currently following an option's policy, return what it selects.
        else:
            return self.executing_options[-1].policy(state)

    def run_agent(
        self, num_epochs, epoch_length, render_interval: int = 0, test_interval: int = 0, test_length: int = 0
    ) -> List[float]:
        """
        Trains the agent for a given number of episodes.

        Args:
            num_epochs (int): The number of epochs to train the agent for.
            epoch_length (int): How many time-steps each epoch should last for.
            render_interval (int, optional): How often (in time-steps) to call the environement's render function, in time-steps. Zero by default, disabling rendering.
            test_interval (int, optional): How often (in epochs) to evaluate the greedy policy learned by the agent. Zero by default, in which case training performance is returned.
            test_length (int, optional): How long (in time-steps) to test the agent for. Zero by default, in which case the agent is tested for one epoch.

        Returns:
            List[float]: A list containing floats representing the rewards earned by the agent each time-step.
        """

        # Set the time-step limit.
        num_time_steps = num_epochs * epoch_length

        episode_rewards = []
        episode = 0
        time_steps = 0

        while time_steps < num_time_steps:
            episode_rewards.append([])

            # Initialise initial state variables.
            state = self.env.reset()
            terminal = False

            if render_interval > 0:
                self.env.render()

            while not terminal:
                selected_option = self.select_action(state)

                # Handle if the selected option is a higher-level option.
                if isinstance(selected_option, Option):
                    self.executing_options.append(copy(selected_option))
                    self.executing_options_states.append([deepcopy(state)])
                    self.executing_options_rewards.append([])

                # Handle if the selected option is a primitive action.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)

                    # Render, if we need to.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    state = deepcopy(next_state)
                    episode_rewards[episode].append(reward)

                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(deepcopy(next_state))
                        self.executing_options_rewards[i].append(reward)

                    # Terminate any options which need terminating this time-step.
                    while self.executing_options and self._roll_termination(self.executing_options[-1], next_state):
                        if self.executing_options[-1] not in self.env.exploration_options:
                            # Perform a macro-q learning update for the terminating option.
                            self.macro_q_learn(
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                self.executing_options[-1],
                                self.n_step_updates,
                            )
                            # Perform an intra-option learning update for the terminating option.
                            self.intra_option_learn(
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                self.executing_options[-1],
                                self.executing_options[-2] if len(self.executing_options) > 1 else None,
                                self.n_step_updates,
                            )
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()
                        self.executing_options.pop()

                # If we have been training for more than the desired number of time-steps, terminate.
                if (time_steps > num_time_steps) and (num_time_steps > 0):
                    terminal = True

                # Handle if the current state is terminal.
                if terminal:
                    while len(self.executing_options) > 0:
                        if self.executing_options[-1] not in self.env.exploration_options:
                            # Perform a macro-q learning update for the topmost option.
                            self.macro_q_learn(
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                self.executing_options[-1],
                                self.n_step_updates,
                            )
                            # Perform an intra-option learning update for the topmost option.
                            self.intra_option_learn(
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                self.executing_options[-1],
                                self.executing_options[-2] if len(self.executing_options) > 1 else None,
                                self.n_step_updates,
                            )
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()
                        self.executing_options.pop()

            episode += 1
        gc.collect()

        return episode_rewards

    def _discounted_return(self, rewards: List[float], gamma: float) -> float:
        # Computes the discounted reward given an ordered list of rewards, and a discount factor.
        num_rewards = len(rewards)

        # Fill an array with gamma^index for index = 0 to index = num_rewards - 1.
        gamma_exp = np.power(np.full(num_rewards, gamma), np.arange(0, num_rewards))

        # Element-wise multiply and then sum array.
        discounted_sum_of_rewards = np.sum(np.multiply(rewards, gamma_exp))

        return discounted_sum_of_rewards

    def _roll_termination(self, option: "Option", state: Hashable):
        # Rolls on whether or not the given option terminates in the given state.
        # Will work with stochastic and deterministic termination functions.
        if random.random() > option.termination(state):
            return False
        else:
            return True
