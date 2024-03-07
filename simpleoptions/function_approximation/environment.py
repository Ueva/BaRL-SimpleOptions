import gymnasium as gym

from typing import List, Dict, Tuple, Hashable
from copy import copy
from abc import ABC, abstractmethod

from simpleoptions.option import BaseOption


class ApproxBaseEnvironment(ABC):
    def __init__(self):
        self.options = set()
        self.exploration_options = set()
        self.current_state = None

    @abstractmethod
    def reset(self, state: Hashable = None) -> Tuple[Hashable, Dict]:
        pass

    @abstractmethod
    def step(self, action: Hashable, state: Hashable = None) -> Tuple[Hashable, float, bool, bool, dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def get_state_space(self) -> gym.spaces.Space:
        pass

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        pass

    def get_available_options(self, state: Hashable, exploration: bool = False) -> List["BaseOption"]:
        """
        Returns the list of options available in the given state.

        Args:
            state (Hashable, optional): The state to check option availability in. Defaults to None, in which case the current state is used.
            exploration (bool, optional): Whether to include options available only for exploration. Defaults to False.

        Returns:
            List[BaseOption]: The set of options available in the given state.
        """
        if state is None:
            state = self.current_state

        # By definition, no options are available in a terminal state.
        if self.is_state_terminal(state):
            return []

        available_options = [option for option in self.options if option.initiation(state)]

        if exploration:
            available_options.extend([option for option in self.exploration_options if option.initiation(state)])

        return available_options

    def set_options(self, new_options: List["BaseOption"], exploration: bool = False, append: bool = False) -> None:
        """
        Sets or updates the set of options available in this environment.

        Args:
            new_options (List[BaseOption]): The new set of options.
            exploration (bool, optional): Whether to update the set of options available only for exploration. Defaults to False.
            append (bool, optional): Whether to append the given options to the current set of options, instead of replacing them. Defaults to False.
        """
        if not append:
            if not exploration:
                self.options = set(copy.copy(new_options))
            else:
                self.exploration_options = set(copy.copy(new_options))
        else:
            if not exploration:
                self.options.update(copy.copy(new_options))
            else:
                self.exploration_options.update(copy.copy(new_options))

        assert self.options.isdisjoint(self.exploration_option)

        # Update the action space to be a discrete set of all available options.
        self.action_space = gym.spaces.Discrete(len(self.options))

        # Maintain a mapping from action indices to options, so we can map network outputs to options.
        self.action_to_option = {i: option for i, option in enumerate(self.options)}
        # TODO: Handle the case where we're updating the set of options. Some options may already exist in the mapping,
        # some options may have been removed from the mapping, and some options may not yet exist in the mapping.


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
        return self.env.unwrapped
