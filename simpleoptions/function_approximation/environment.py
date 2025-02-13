import random

import numpy as np
import gymnasium as gym

from typing import List, Dict, Tuple, Hashable
from abc import ABC, abstractmethod

from simpleoptions.option import BaseOption


class ApproxBaseEnvironment(ABC):
    """
    Abstract base class for environments requiring function approximation. Very similar to OpenAI Gym environments.
    To implement your own environment, you should subclass this class and implement its abstract methods.
    Additionally, you should use gymnasium `spaces` to define state and action spaces.
    """

    def __init__(self, render_mode: str = "human"):
        self.render_mode = render_mode

        # Create initial option sets.
        self._options = list()
        self._exploration_options = list()
        self.option_space = None

        self.current_state = None

    @abstractmethod
    def reset(self, state: Hashable = None) -> Tuple[Hashable, Dict]:
        """
        This method resets the environment, either to a specified state or to a random initial state.

        Args:
            state (Hashable, optional): The state to reset the environment to. Defaults to None, in which case the environment is reset to a random initial state.

        Returns:
            Tuple[Hashable, Dict]: A tuple containing the initial state of the environment and a dictionary containing any additional information.
        """
        pass

    @abstractmethod
    def step(self, action: Hashable, state: Hashable = None) -> Tuple[Hashable, float, bool, bool, dict]:
        """
        This method implements the one-step dynamics of the environment. Simulates one step of interaction in the environment,
        transitioning to a new state, recieving a reward, and determining whether the episode has terminated or truncated.

        Args:
            action (Hashable): A primitive action to execute in the environment (from self.get_action_space()).
            state (Hashable, optional): Optionally, the state to execute the action in. Defaults to None, in which case the action is executed in the current state.

        Returns:
            Tuple[Hashable, float, bool, bool, dict]: A tuple containing the next state, reward, whether the episode has terminated, whether the episode has been truncated, and a dictionary containing any additional information.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """
        Display a representation of the environment's current state.
        """
        pass

    def seed(self, random_seed: int) -> None:
        """
        Seed the environment's random number generator(s).

        Args:
            random_seed (int): The random seed to use for random number generation.
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

    @abstractmethod
    def close(self) -> None:
        """
        Cleanly terminates all environment-related processes and releases any currently-held resources (e.g., closing any renderers).
        """
        pass

    @abstractmethod
    def get_state_space(self) -> gym.spaces.Space:
        """
        Returns a gymnasium space representing the environment's state space.

        Returns:
            gym.spaces.Space: The environment's state space.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Discrete:
        """
        Returns a discrete gymnasium space representing the environment's action space.

        Returns:
            gym.spaces.Discrete: The environment's action space.
        """
        pass

    def get_option_space(self) -> gym.spaces.Discrete:
        """
        Returns a discrete gymnasium space representing the environment's option space.

        Returns:
            gym.spaces.Discrete: The environment's option space.
        """
        return self.option_space

    def get_available_options(
        self, state: Hashable, exploration: bool = False, get_indices: bool = False
    ) -> List["BaseOption"]:
        """
        Returns the list of options available in the given state.

        Args:
            state (Hashable, optional): The state to check option availability in. Defaults to None, in which case the current state is used.
            exploration (bool, optional): Whether to include options available only for exploration. Defaults to False.
            get_indices (bool, optional): Whether to return the indices of the selected options instead of the options themselves. Defaults to False. Not compatible with exploration=True.

        Returns:
            List[BaseOption]: The set of options available in the given state.
        """
        if exploration and get_indices:
            # Exploration options aren't part of the option space - they aren't aviailable for the
            # agent to select directly. So, they are not given indices in the option space.
            raise ValueError(
                "Cannot return the indices of exploration options. `exploration` and `get_indices` are mutually exclusive."
            )

        # An option is available if the given state satisfies its initiation set.
        available_options = [option for option in self._options if option.initiation(state)]
        if exploration:
            available_options.extend([option for option in self._exploration_options if option.initiation(state)])

        if not get_indices:
            return available_options
        else:
            return [self.option_to_index[option] for option in available_options]

    def get_available_option_mask(self, state: Hashable) -> np.ndarray:
        """
        Returns a mask of available options in the given state.

        Args:
            state (Hashable): The state to check option availability in.

        Returns:
            np.ndarray: A binary mask indicating which options are available in the given state.
        """
        mask = np.zeros(self.option_space.n, dtype=np.int32)
        available_options = self.get_available_options(state, get_indices=True)
        mask[available_options] = 1

        return mask

    def set_options(self, new_options: List["BaseOption"], append: bool = False) -> None:
        """
        Sets or updates the set of options available in this environment.

        Args:
            new_options (List[BaseOption]): The new set of options.
            append (bool, optional): Whether to append the given options to the current set of options, instead of replacing them. Defaults to False.
        """

        # Remove duplicates from the list of new options.
        new_options = list(set(new_options))

        # Replace the existing set of options with a new set of options.
        if not append:
            self._options = []
            self._options.extend(new_options)
        # Append the new set of options to the existing set of options.
        else:
            existing_options = set(self._options)
            options_to_add = [option for option in new_options if option not in existing_options]
            self._options.extend(options_to_add)

        # Update the action space to be a discrete set of all available options.
        self.option_space = gym.spaces.Discrete(len(self._options))

        # Maintain mappings between option indices and options, so we can map network outputs to options.
        self.index_to_option = {i: option for i, option in enumerate(self._options)}
        self.option_to_index = {option: i for i, option in self.index_to_option.items()}

        # ? Consider keeping track of indicies used for options that no longer exist in the option space, and
        # ? assign new indices to new options. Really, the user should be responsible for this if they decide to
        # ? change the set of available options during training, but it might be worth supporting here.

    def set_exploration_options(self, new_options: List["BaseOption"], append: bool = False) -> None:
        """
        Sets or updates the set of exploration options available in this environment.

        Args:
            new_options (List[BaseOption]): The new set of exploration options.
            append (bool, optional): Whether to append the given options to the current set of exploration options, instead of replacing them. Defaults to False.
        """

        # Remove duplicates from the list of new options.
        new_options = list(set(new_options))

        # Replace the existing set of exploration options with a new set of options.
        if not append:
            self._exploration_options = new_options
        # Append the new set of exploration options to the existing set of exploration options.
        else:
            existing_options = set(self._exploration_options)
            options_to_add = [option for option in new_options if option not in existing_options]
            self._exploration_options.extend(options_to_add)

        # Note: Exploration options are not included in the option space, because they are not directly selectable by the agent.
        # So, we do not need to keep track of mappings between exploration options and indices.


class GymWrapper(ApproxBaseEnvironment):
    """
    A wrapper for OpenAI Gym environments, to make them compatible with the ApproxBaseEnvironment interface.
    """

    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env

    def reset(self, state: Hashable = None) -> Tuple[Hashable, Dict]:
        return self.env.reset()

    def step(self, action: Hashable, state: Hashable = None) -> Tuple[Hashable, float, bool, bool, dict]:
        return self.env.step(action)

    def render(self) -> None:
        return self.env.render()

    def seed(self, random_seed: int) -> None:
        return self.env.seed(random_seed)

    def close(self) -> None:
        return self.env.close()

    def get_state_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    def get_action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def reward_range(self) -> Tuple[float, float]:
        return self.env.reward_range

    @property
    def unwrapped(self) -> gym.Env:
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env
