import gc
import math
import random
import statistics
import numpy as np
from numpy.random import Generator as RNG

from copy import copy
from collections import defaultdict
from typing import Tuple, Hashable, List, Union, DefaultDict

from simpleoptions.option import BaseOption
from simpleoptions.environment import BaseEnvironment
from simpleoptions.utils.math import discounted_return


class OptionAgent:
    """
    An agent which acts in a given environment, learning using the Macro-Q learning
    and intra-option learning algorithms.
    """

    def __init__(
        self,
        env: "BaseEnvironment",
        test_env: "BaseEnvironment" = None,
        epsilon: float = 0.15,
        macro_alpha: float = 0.2,
        intra_option_alpha: float = 0.2,
        gamma: float = 1.0,
        default_action_value=0.0,
        n_step_updates=False,
        rng: RNG = None,
        *args,
        **kwargs,
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
        self.epsilon = epsilon
        self.gamma = gamma
        self.macro_q_alpha = macro_alpha
        self.intra_option_alpha = intra_option_alpha
        self.executing_options = []
        self.executing_options_states = []
        self.executing_options_rewards = []
        self.n_step_updates = n_step_updates
        self.rng = rng if rng else random

        self.env = env
        self.test_env = test_env if test_env is not None else None

        self.evaluation_log = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.training_log = defaultdict(list)

    def macro_q_learn(
        self,
        state_trajectory: List[Hashable],
        rewards: List[float],
        option: "BaseOption",
        n_step=False,
    ) -> None:
        """
        Performs Macro Q-Learning updates along the given trajectory for the given Option.

        Args:
            state_trajectory (List[Hashable]): The list of states visited each time-step while the option was executing.
            rewards (List[float]): The list of rewards earned each time-step while the Option was executing.
            option (Option): The option to perform an update for.
            n_step (bool): Whether or not to perform n-step updates. Defaults to False, performing one-step updates.
        """
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

        for i in range(len(state_trajectory) - 1):
            initiation_state = state_trajectory[i]

            old_value = self.q_table[(hash(initiation_state), hash(option))]

            # Compute discounted sum of rewards.
            discounted_sum_of_rewards = discounted_return(rewards[i:], self.gamma)

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
                discounted_sum_of_rewards + math.pow(self.gamma, len(rewards) - i) * max(q_values) - old_value
            )

            # If we're not performing n-step updates, exit after the first iteration.
            if not n_step:
                break

    def intra_option_learn(
        self,
        state_trajectory: List[Hashable],
        rewards: List[float],
        executed_option: BaseOption,
        higher_level_option: Union["BaseOption", None] = None,
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

        termination_state = state_trajectory[-1]

        for i in range(len(state_trajectory) - 1):
            initiation_state = state_trajectory[i]

            # We perform an intra-option update for all other options which select executed_option in this state.
            for other_option in self.env.get_available_options(initiation_state):
                if (
                    (hash(other_option) != hash(higher_level_option) or higher_level_option is None)
                    and hash(other_option.policy(initiation_state)) == hash(executed_option)
                    # and other_option.initiation(initiation_state) # This check is already handled in env.get_available_options!
                ):
                    old_value = self.q_table[(hash(initiation_state), hash(other_option))]

                    # Compute discounted sum of rewards.
                    discounted_sum_of_rewards = discounted_return(rewards[i:], self.gamma)

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
                        + math.pow(self.gamma, len(rewards) - i) * (next_q_continues + next_q_terminates)
                        - old_value
                    )

            # If we're not performing n-step updates, exit after the first iteration.
            if not n_step:
                break

    def select_action(
        self, state: Hashable, executing_options: List["BaseOption"], test: bool = False
    ) -> Union[BaseOption, Hashable, None]:
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
        if len(executing_options) == 0:
            # Random Action.
            if not test and self.rng.random() < self.epsilon:
                available_options = self.env.get_available_options(state, exploration=True)
                return self.rng.choice(available_options)
            # Best Action.
            else:
                available_options = self.env.get_available_options(state, exploration=False)
                # Find Q-values of available options.
                q_values = [self.q_table[(hash(state), hash(o))] for o in available_options]

                # Return the option with the highest Q-value, breaking ties randomly.
                return available_options[
                    self.rng.choice([idx for idx, q_value in enumerate(q_values) if q_value == max(q_values)])
                ]
        # If we are currently following an option's policy, return what it selects.
        else:
            return executing_options[-1].policy(state, test)

    def run_agent(
        self,
        num_epochs: int,
        epoch_length: int,
        render_interval: int = 0,
        test_interval: int = 0,
        test_length: int = 0,
        test_runs: int = 10,
        verbose_logging: bool = True,
        episodic_eval: bool = False,
    ) -> Tuple[DefaultDict, DefaultDict | None]:
        """
        Trains the agent for a given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the agent for.
            epoch_length (int): How many time-steps each epoch should last for.
            render_interval (int, optional): How often (in time-steps) to call the environement's render function, in time-steps. Zero by default, disabling rendering.
            test_interval (int, optional): How often (in epochs) to evaluate the greedy policy learned by the agent. Zero by default, in which case training performance is returned.
            test_length (int, optional): How long (in time-steps) to test the agent for. Zero by default, in which case the agent is tested for one epoch.
            test_runs (int, optional): How many test runs to perform each test_interval.
            verbose_logging (bool, optional): Whether to log all information about each time-step, instead of just rewards. Defaults to True.

        Returns:
            Tuple[DefaultDict, DefaultDict | None]: A tuple of dictionaries, (training_logs, evaluation_logs), containing data logs of training and evaluation.
        """

        # Set the time-step limit.
        num_time_steps = num_epochs * epoch_length

        # If we are testing the greedy policy separately, make a separate copy of
        # the environment to use for those tests. Also initialise variables for
        # tracking test performance.
        training_rewards = [None for _ in range(num_time_steps)]

        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]

            # Check that a test environment has been provided - if not, raise an error.
            if self.test_env is None:
                raise RuntimeError("No test_env has been provided specified.")
        else:
            evaluation_rewards = []

        episode = 0
        time_steps = 0

        while time_steps < num_time_steps:
            # Initialise initial state variables.
            state = self.env.reset()
            terminal = False
            if render_interval > 0:
                self.env.render()

            while not terminal:
                selected_option = self.select_action(state, self.executing_options)

                # Handle if the selected option is a higher-level option.
                if isinstance(selected_option, BaseOption):
                    self.executing_options.append(copy(selected_option))
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])

                # Handle if the selected option is a primitive action.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.env.step(selected_option)
                    # Logging
                    training_rewards[time_steps - 1] = reward
                    if verbose_logging:
                        transition = {
                            "state": state,
                            "next_state": next_state,
                            "reward": reward,
                            "terminal": terminal,
                            "active_options": [str(option) for option in self.executing_options],
                        }
                        for key, value in transition.items():
                            self.training_log[key].append(value)

                    # Render, if we need to.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    state = next_state

                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(next_state)
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

                    # If we are testing the greedy policy learned by the agent separately,
                    # and it is time to test it, then test it.
                    if test_interval > 0 and time_steps % test_interval_time_steps == 0:
                        evaluation_rewards[(time_steps - 1) // test_interval_time_steps] = self.test_policy(
                            test_length,
                            test_runs,
                            time_steps // test_interval_time_steps,
                            allow_exploration=False,
                            verbose_logging=verbose_logging,
                            episodic_eval=episodic_eval,
                        )

                # If we have been training for more than the desired number of time-steps, terminate.
                if time_steps >= num_time_steps:
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

        if verbose_logging:
            training_log = self.training_log
            evaluation_log = self.evaluation_log if self.evaluation_log else None
            return training_log, evaluation_log
        else:
            training_log = [sum(training_rewards[i * epoch_length : (i + 1) * epoch_length]) for i in range(num_epochs)]
            evaluation_log = evaluation_rewards if evaluation_rewards else None
            return training_log, evaluation_log

    def test_policy(
        self,
        test_length: int,
        test_runs: int,
        eval_number: int,
        allow_exploration: bool = False,
        verbose_logging: bool = True,
        episodic_eval: bool = False,
    ):
        """Evaluates the agents current policy.

        Args:
            test_length (int): Number of timesteps to evaluate agent over,
                                    or episode cut off steps if `episodic_eval`.
            test_runs (int): Number of evaluations to perform,
                                    or number of episodes to evaluate if `episodic_eval`.
            eval_number (int): Unique sequential identifier of current evaluation.
            allow_exploration (bool, optional): Toggle between epsilon-greedy (True) and epsilon policies (False). Defaults to False.
            verbose_logging (bool, optional): Enables detailed logging of evaluation. Defaults to True.
            episodic_eval (bool, optional): Toggle between episodic (True) and fixed timestep (False) evaluations. Defaults to False.

        Returns:
            _type_: _description_
        """

        test_total_rewards = [None for _ in range(test_runs)]

        for test_run in range(test_runs):
            time_steps = 0
            cumulative_reward = 0
            state = self.test_env.reset()
            executing_options = []
            terminal = False
            while time_steps < test_length and not (terminal and episodic_eval):
                selected_option = self.select_action(state, executing_options, test=not allow_exploration)

                # Handle if the selected option is a higher-level option.
                if isinstance(selected_option, BaseOption):
                    executing_options.append(copy(selected_option))

                # Handle if the selected option is a primitive action.
                else:
                    time_steps += 1
                    next_state, reward, terminal, __ = self.test_env.step(selected_option)

                    # Logging
                    cumulative_reward += reward
                    if verbose_logging:
                        transition = {
                            "state": state,
                            "next_state": next_state,
                            "reward": reward,
                            "terminal": terminal,
                            "active_options": [str(option) for option in executing_options],
                        }
                        for key, value in transition.items():
                            self.evaluation_log[f"evaluation_{eval_number}"][f"run_{test_run+1}"][key].append(value)

                    # Reset environment and continue evaluation run.
                    if terminal and not episodic_eval:
                        state = self.test_env.reset()
                        executing_options = []
                        terminal = False
                    # Continue evaluation run
                    else:
                        state = next_state
                        # Terminate any options which need terminating this time-step.
                        while executing_options and self._roll_termination(executing_options[-1], next_state):
                            executing_options.pop()

            test_total_rewards[test_run] = cumulative_reward

        return statistics.mean(test_total_rewards)

    def _roll_termination(self, option: "BaseOption", state: Hashable):
        # Rolls on whether or not the given option terminates in the given state.
        # Will work with stochastic and deterministic termination functions.
        if self.rng.random() > option.termination(state):
            return False
        else:
            return True
