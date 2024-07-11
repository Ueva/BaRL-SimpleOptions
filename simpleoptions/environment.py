import copy
import random

import networkx as nx

from typing import List, Set
from typing import Hashable, Tuple
from abc import ABC, abstractmethod

from simpleoptions.option import BaseOption


class BaseEnvironment(ABC):
    """
    Abstract base class for environments, similar to OpenAI Gym environments.
    You should implement one-step dynamics for your environment, a "reset" function to
    initialise a the envionrment before starting an episode, and a function for returning
    the options available to the agent in a given state.
    """

    def __init__(self):
        """
        Constructs a new environment object.
        """
        self._options = set()
        self.exploration_options = set()
        self._option_availability_maps = {}
        self._exploration_option_availability_maps = {}
        self.current_state = None

    @abstractmethod
    def reset(self, state: Hashable = None) -> Hashable:
        """
        This method initialises, or reinitialises, the environment prior to starting a new episode.
        It returns an initial state.

        Arguments:
            state (Hashable, optional) -- The state to reset the environment to. Defaults to None, and resets the environment to a random initial state.

        Returns:
            Hashable -- An initial environmental state.
        """
        pass

    @abstractmethod
    def step(self, action: Hashable, state: Hashable = None) -> Tuple[Hashable, float, bool, dict]:
        """
        This method implements the one-step transition dynamics of the environment. Given an action,
        the environment transitions to some next state accordingly.

        Arguments:
            action (Hashable) -- The action for the agent to take in the current environmental state.
            state (Hashable, optional) -- The state to take the given action in. Defaults to None, and uses the current environmental state.

        Returns:
            Hashable -- The next environmental state.
            float -- The reward earned by the agent by taking the given action in the current state.
            bool -- Whether or not the new state is terminal.
            dict -- A dictionary containing information about the current episode.

        next_state, reward, terminal, info -- ENSURE ORDER IS CORRECT!!!
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
    def get_state_space(self) -> Set[Hashable]:
        """
        Returns a set containing all of the possible states in this environment.

        Returns:
            Set[Hashable]: All possible states in this environment.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> Set[Hashable]:
        """
        Returns a set containing all of the primitive actions available in this environment. A subset of
        the actions in this list will be available in each individual state.

        Returns:
            Set[Hashable]: All possible primitive actions available in this environment.

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

    def get_option_space(self) -> Set["BaseOption"]:
        """
        Returns a set containing all of the options available in this environment.

        Returns:
            Set[BaseOption]: All possible options available in this environment.
        """
        return self._options

    def get_available_options(self, state: Hashable, exploration=False) -> List["BaseOption"]:
        """
        This method returns the options (primitive options + subgoal options) which are available to the
        agent in the given environmental state.

        Arguments:
            state (Hashable) -- The state to return the available options for. Defaults to None, and uses the current environmental state.
            exploration (bool) -- Whether to include options that are only available for exploration.

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
            available_options = copy.copy(self._option_availability_maps.get(state, list()))
            if exploration:
                available_options.extend(copy.copy(self._exploration_option_availability_maps.get(state, list())))

            return available_options

    def set_options(self, new_options: List["BaseOption"], append: bool = False) -> None:
        """
        Sets the set of options available in this environment.
        By default, replaces the current list of available options. If you wish to extend the
        list of currently avaialble options, set the `append` parameter to `True`.

        Args:
            new_options (List[BaseOption]): The list of options to make avaialble.
            append (bool, optional): Whether to append the new options to the current set of options. Defaults to False.
        """
        if not append:
            self._options = set(copy.copy(new_options))
        else:
            self._options.update(copy.copy(new_options))

        self._option_availability_maps = {}
        for state in self.get_state_space():
            for option in self._options:
                if option.initiation(state):
                    self._option_availability_maps[state] = self._option_availability_maps.get(state, list())
                    self._option_availability_maps[state].append(option)

    def set_exploration_options(self, new_options: List["BaseOption"], append: bool = False) -> None:
        """
        Sets the set of options available solely for exploration in this environment.
        By default, replaces the current list of available options. If you wish to extend the
        list of currently avaialble options, set the `append` parameter to `True`.
        Args:
            new_options (List[BaseOption]): The list of options to make avaialble.
            append (bool, optional): Whether to append the new options to the current set of exploration options. Defaults to False.
        """
        if not append:
            self.exploration_options = set(copy.copy(new_options))
        else:
            self.exploration_options.update(copy.copy(new_options))

        self._exploration_option_availability_maps = {}
        for state in self.get_state_space():
            for exploration_option in self.exploration_options:
                if exploration_option.initiation(state):
                    self._exploration_option_availability_maps[state] = self._exploration_option_availability_maps.get(
                        state, list()
                    )
                    self._exploration_option_availability_maps[state].append(exploration_option)

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
    def get_successors(
        self, state: Hashable = None, actions: List[Hashable] = None
    ) -> List[Tuple[Tuple[Hashable, float], float]]:
        """
        Returns a list of next-states and rewards that can be reached by taking an action in the given state, and the probabilities
        of transitioning to each of them.
        If no state is specified, a list of successor states for the current state will be returned.

        Args:
            state (Hashable, optional): The state to return successors for. Defaults to None (i.e. current state).
            actions (List[Hashable], optional): The actions to test in the given state when searching for successors. Defaults to None (i.e. tests all available actions).

        Returns:
            List[Tuple[Hashable, float, float]]: A list of next-states and rewards reachable by taking an action in the given state, and the probabilities of transitioning to them.
        """
        pass

    def generate_interaction_graph(self, directed=True, weighted=False) -> "nx.DiGraph":
        """
        Returns a NetworkX DiGraph representing the state-transition graph for this environment.

        Arguments:
            directed (bool, optional): Whether the state-transition graph should be directed. Defaults to True.
            weighted (bool, optional): Whether the state-transition graph should be weighted. Defaults to False.

        Raises:
            ValueError: If weighted is True and directed is False. Weighted graphs must be directed.

        Returns:
            nx.DiGraph: A NetworkX DiGraph representing the state-transition graph for this environment. Nodes are states, edges are possible transitions, edge weights are one.
        """

        if weighted and not directed:
            raise ValueError("Weighted graphs must be directed.")

        if not weighted:
            return self._generate_interaction_graph_unweighted(directed=directed)
        else:
            return self._generate_interaction_graph_weighted()

    def _generate_interaction_graph_unweighted(self, directed=False) -> "nx.DiGraph":
        # Generates a list of all reachable states, starting the search from the environment's initial states.
        states = []
        current_successor_states = self.get_initial_states()

        # Brute force construction of the state-transition graph. Starts with initial states,
        # then tries to add possible successor states until no new successor states can be added.
        while not len(current_successor_states) == 0:
            next_successor_states = []
            for successor_state in current_successor_states:
                if not successor_state in states:
                    states.append(successor_state)

                    if not self.is_state_terminal(successor_state):
                        new_successors = self.get_successors(successor_state)
                        for (new_successor_state, _), _ in new_successors:
                            next_successor_states.append(new_successor_state)

            current_successor_states = copy.deepcopy(next_successor_states)

        # Build state-transition graph.
        if directed:
            stg = nx.DiGraph()
        else:
            stg = nx.Graph()
        for state in states:
            # Add node for state.
            stg.add_node(state)

            # Add directed edge between node and its successors.
            successors = self.get_successors(state)
            for (successor_state, _), _ in successors:
                stg.add_node(successor_state)
                stg.add_edge(state, successor_state)

        return stg

    def _generate_interaction_graph_weighted(self) -> "nx.DiGraph":
        # Generates a list of all reachable states, starting the search from the environment's initial states.
        states = []
        current_successor_states = self.get_initial_states()

        # Brute force construction of the state-transition graph. Starts with initial states,
        # then tries to add possible successor states until no new successor states can be added.
        while not len(current_successor_states) == 0:
            next_successor_states = []
            for successor_state in current_successor_states:
                if not successor_state in states:
                    states.append(successor_state)

                    if not self.is_state_terminal(successor_state):
                        new_successors = self.get_successors(successor_state)
                        for (new_successor_state, _), _ in new_successors:
                            next_successor_states.append(new_successor_state)

            current_successor_states = copy.deepcopy(next_successor_states)

        # Build state-transition graph with multiple edges between each pair of nodes.
        stg = nx.DiGraph()
        for state in states:
            # Add node for state.
            stg.add_node(state)

            # Add directed edge between node and its successors.
            successors = self.get_successors(state)
            for (successor_state, _), transition_prob in successors:

                stg.add_node(successor_state)
                if stg.has_edge(state, successor_state):
                    stg[state][successor_state]["weight"] += transition_prob
                else:
                    stg.add_edge(state, successor_state, weight=transition_prob)

        return stg


class TransitionMatrixBaseEnvironment(BaseEnvironment):
    def __init__(self, deterministic: bool = True):
        self.deterministic = deterministic
        super().__init__()

        self.transition_matrix = self._compute_transition_matrix()

    def _compute_transition_matrix(self):
        # We want to create a dictionary representing the transition matrix for this environment.
        # The dictionary should be keyed by state-action pair, and contain a list of (next_state, probability) tuples.

        transition_matrix = {}

        # We can iterate through each state in the state space using the get_state_space() method.
        for state in self.get_state_space():
            for action in self.get_available_actions(state=state):
                transition_matrix[(state, action)] = self.get_successors(state=state, actions=[action])

        return transition_matrix

    def step(self, action, state=None):
        # If the environment is deterministic, we can simply return the next state and reward.
        if self.deterministic:
            (next_state, reward), _ = self.transition_matrix[(state, action)][0]
        # Otherwise, we sample a next state and reward based on the transition matrix.
        else:
            outcomes, probabilities = zip(*self.transition_matrix[(state, action)])
            (next_state, reward) = random.choices(outcomes, probabilities, k=1)[0]

        # Determine whether the next state is terminal.
        terminal = self.is_state_terminal(next_state)

        return next_state, reward, terminal, {}
