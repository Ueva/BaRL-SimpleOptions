from typing import Hashable

from abc import ABC, abstractmethod


class BaseOption(ABC):
    _class_id_counter = 0

    """
    Interface for a behaviour represented using the options framework in hierarchical reinforcement learning.
    """

    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._class_id = BaseOption._class_id_counter
        BaseOption._class_id_counter += 1

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
        raise NotImplementedError("BaseOption is an abstract class and should not be instantiated directly.")


class PseudoRewardOption(BaseOption):
    """
    Interface for a behaviour, represented using the options framework in hierarchical
    reinforcement learning, which acts to maximise a specific pseudo-reward signal.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def pseudo_reward(self, state: Hashable, action: Hashable, next_state: Hashable) -> float:
        """
        Returns this option's pseudo-reward for transitioning from `state` to `next_state` via `action`.
        It is this pseudo-reward signal that this option's policy should aim to maximise.

        Args:
            state (Hashable): The state being transitioned *from*.
            action (Hashable): The action being transitioned *via*.
            next_state (Hashable): The state being transitioned *to*.

        Returns:
            float: the pseudo-reward signal for transitioning from `state` to `next_state` via `action`.
        """
        pass
