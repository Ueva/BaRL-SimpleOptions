from typing import Hashable

from abc import ABC, abstractmethod


class BaseOption(ABC):
    """
    Interface for a hierarchical reinforcement learning option.
    """

    def __init__(self):
        pass

    @abstractmethod
    def initiation(self, state: Hashable) -> bool:
        """
        Returns whether or not a given state is in this option's initation set.

        Arguments:
            state {Hashable} -- The state whose membership to the initiation set is to be tested.

            Returns:
                bool -- [description]
        """
        pass

    @abstractmethod
    def policy(self, state: Hashable, test: bool = False) -> Hashable:
        """
        Returns the action specified by this option's policy for a given state.

        Arguments:
            state {Hashable} -- The environmental state in which the option chooses an action in.
            test [bool] --  When True the option follows a greedy-policy, for evaluation. Defaults to False.

        Returns:
            action [Hashable] -- The action specified by the option's policy in this state.
        """
        pass

    @abstractmethod
    def termination(self, state: Hashable) -> float:
        """
        Returns the probability that the option terminates in the given state.

        Arguments:
            state {Hashable} -- The state in which to test for termination.

        Returns:
            float -- The probability that this option terminates in this state.
        """
        pass

    @abstractmethod
    def __hash__(self):
        pass
