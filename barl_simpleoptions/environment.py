import networkx as nx

from typing import List
from typing import Hashable, Tuple

from abc import ABC, abstractmethod

from barl_simpleoptions.option import Option


class BaseEnvironment(ABC):
    """
    Abstract base class for environments, similar to OpenAI Gym environments.
    You should implement one-step dynamics for your environment, a "reset" function to
    initialise a the envionrment before starting an episode, and a function for returning
    the options available to the agent in a given state.
    """

    def __init__(self, options: List["Option"] = []):
        """
        Constructs a new environment object.

        Args:
            options (List["Option"], optional): A set of options to initialise this environment with. Defaults to an empty list [].
        """
        self.options = options
        self.current_state = None

    @abstractmethod
    def step(self, action: Hashable) -> Tuple[Hashable, float, bool, dict]:
        """
        This method implements the one-step transition dynamics of the environment. Given an action,
        the environment transitions to some next state accordingly.

        Arguments:
            action (Hashable) -- The action for the agent to take in the current environmental state.

        Returns:
            Hashable -- The next environmental state.
            float -- The reward earned by the agent by taking the given action in the current state.
            bool -- Whether or not the new state is terminal.
            dict -- A dictionary containing information about the current episode.

        next_state, reward, terminal, info -- ENSURE ORDER IS CORRECT!!!
        """
        pass

    @abstractmethod
    def reset(self) -> Hashable:
        """
        This method initialises, or reinitialises, the environment prior to starting a new episode.
        It returns an initial state.

        Returns:
            Hashable -- An initial environmental state.
        """
        pass

    @abstractmethod
    def render(self, mode: str = "human") -> None:
        """
        Displays a representation of the environment's current state.

        Args:
            mode (str, optional): Optional, used to specify the rendering method for environments with multiple rendering modes.

        Returns:
            [type]: [description]
        """
        pass

    @abstractmethod
    def close(self):
        """
        Cleanly terminates all environment-related process (e.g. closing any renderers).
        """

    @abstractmethod
    def get_available_actions(self, state: Hashable = None) -> List[Hashable]:
        """
        This method returns a list of primitive actions (NOT options) which are
        available to the agent in this state.

        Args:
            state (Hashable): The state to return available primitve actions for. Defaults to None, and uses the current environmental state.

        Returns:
            List[Hashable]: The list of options available in this state.
        """
        pass

    def get_available_options(self, state: Hashable) -> List["Option"]:
        """
        This method returns the options (primitive options + subgoal options) which are available to the
        agent in the given environmental state.

        Arguments:
            state (Hashable) -- The state to return the available options for. Defaults to None, and uses the current environmental state.

        Returns:
            List[Option] -- The list of options available in this state.
        """
        if state is None:
            state = self.current_state

        # Lists all options (including options corresponding to primitive actions) which have the given state in their initiation sets.
        available_options = [option for option in self.options if option.initiation(state)]

        return available_options

    @abstractmethod
    def is_state_terminal(self, state: Hashable = None) -> bool:
        """
        Returns whether the given state is terminal.

        Args:
            state (Hashable, optional): The state to check terminal status for. Defaults to None, and uses the current environmental state.

        Returns:
            bool: Whether the given state is terminal.
        """
        pass

    @abstractmethod
    def get_initial_states(self) -> List[Hashable]:
        """
        Gets a list of possible initial states for this environment.

        Returns:
            List[Hashable]: A list containing the possible initial states in this environment.
        """
        pass

    @abstractmethod
    def generate_interaction_graph(self) -> "nx.DiGraph":
        """
        Returns a NetworkX DiGraph representing the state-transition graph for this environment.

        Returns:
            nx.DiGraph: A NetworkX DiGraph representing the state-transition graph for this environment. Nodes are states, edges are possible transitions, edges weights are one.
        """
