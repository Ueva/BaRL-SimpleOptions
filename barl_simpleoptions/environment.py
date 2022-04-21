import networkx as nx

from typing import List
from typing import Hashable, Tuple
from abc import ABC, abstractmethod
from copy import deepcopy

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
        self.options = deepcopy(options)
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
        pass

    @abstractmethod
    def get_action_space(self) -> List[Hashable]:
        """
        Returns a list containing the set of primitive actions available in this environment. A subset of
        the actions in this list will be available in each individual state.

        Returns:
            List[Hashable]: All possible primitive actions available in this environment.

        See Also:
            BaseEnvironment.get_available_actions
        """
        pass

    @abstractmethod
    def get_available_actions(self, state: Hashable = None) -> List[Hashable]:
        """
        This method returns a list of primitive actions (NOT options) which are available to
        the agent in this state. Should return some subset of self.get_action_space().

        Args:
            state (Hashable): The state to return available primitve actions for. Defaults to None, and uses the current environmental state.

        Returns:
            List[Hashable]: The list of options available in this state.

        See Also:
            BaseEnvironment.get_action_space
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

        # By definition, no options are available in the terminal state.
        if self.is_state_terminal(state):
            return []
        # Otherwise, options whose initiation set contains the given state are returned.
        else:
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
    def get_successors(self, state: Hashable = None, actions: List[Hashable] = None) -> List[Hashable]:
        """
        Returns a list of states which can be reached by taking an action in the given state.
        If no state is specified, a list of successor states for the current state will be returned.

        Args:
            state (Hashable, optional): The state to return successors for. Defaults to None (i.e. current state).
            actions (List[Hashable], optional): The actions to test in the given state when searching for successors. Defaults to None (i.e. tests all available actions).

        Returns:
            List[Hashable]: A list of states reachable by taking an action in the given state.
        """
        pass

    def generate_interaction_graph(self, directed=False) -> "nx.DiGraph":
        """
        Returns a NetworkX DiGraph representing the state-transition graph for this environment.

        Returns:
            nx.DiGraph: A NetworkX DiGraph representing the state-transition graph for this environment. Nodes are states, edges are possible transitions, edges weights are one.
        """

        # Generates a list of all reachable states, starting the search from the environment's initial states.
        states = []
        current_successor_states = self.get_initial_states()

        # Brute force construction of the state-transition graph. Starts with initial
        # states, then tries to add possible successor states until no new successor states
        # can be added. This can take quite a while for environments with a large state-space.
        while not len(current_successor_states) == 0:
            next_successor_states = []
            for successor_state in current_successor_states:
                if not successor_state in states:
                    states.append(successor_state)

                    if not self.is_state_terminal(successor_state):
                        for new_successor_state in self.get_successors(successor_state):
                            next_successor_states.append(new_successor_state)

            current_successor_states = deepcopy(next_successor_states)

        # Build state-transition graph.
        if directed:
            stg = nx.DiGraph()
        else:
            stg = nx.Graph()
        for state in states:
            # Add node for state.
            stg.add_node(state)

            # Add directed edge between node and its successors.
            for successor_state in self.get_successors(state):
                stg.add_node(successor_state)
                stg.add_edge(state, successor_state)

        return stg
