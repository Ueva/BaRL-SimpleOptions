import os
import math
import json
import random
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

class State(ABC) :
    """
    This abstract class represents a discrete state in a reiforcement learning problem.
    The abstract methods should be implemented as per the requirements and transition 
    dynamics for whatever environment you are working with.
    """

    def __init__(self) :
        pass
    
    @abstractmethod
    def __str__(self) :
        pass # Make sure that each state can be represented uniquely as a hashable string.

    def __repr__(self):
        return str(self)

    @abstractmethod
    def __eq__(self, other_state : State):
        pass # Make sure that you have defined equality between your states correctly.

    def __ne__(self, other_state : State):
        return not self.__eq__(other_state)
    
    @abstractmethod
    def get_available_actions(self) :
        """
        Returns the list of actions available in this state.

        Returns:
            List[Hashable] -- The list of actions available in this state.
        """
        pass

    @abstractmethod
    def take_action(self, action) :
        """
        Returns the list of states which can be arrived at by taking the given action in this state.
        
        Arguments:
            action {Hashable} -- The action to take in this state. Should be a member of the list returned by get_available_actions.

        Returns:
            List[State] -- The list of states which are reachable from this state when taking the specified action.
        """
        pass

    @abstractmethod
    def is_action_legal(self, action) -> bool :
        """
        Returns whether the given action is legal in this state.
        
        Arguments:
            action {Hashable} -- The action to check for legaility in this state.

        Returns:
            bool -- Whether or not the given action is legal in this state.
        """
        pass

    @abstractmethod
    def is_state_legal(self) -> bool :
        """
        Returns whether or not the current state is legal.

        Returns:
            bool -- Whether or not this state is legal.
        """
        pass

    @abstractmethod
    def is_initial_state(self) -> bool :
        """
        Returns whether or not this state is an initial state.

        Returns:
            bool -- Whether or not this state is an initial state.
        """
        pass

    @abstractmethod
    def is_terminal_state(self) -> bool :
        """
        Returns whether or not this is a terminal state.
        
        Returns:
            bool -- Whether or not this state is terminal.
        """
        pass
        
    @abstractmethod
    def get_successors(self) -> List[State] :
        """
        Returns a list of all states which can be reached from this state in one time step.

        Returns:
            List[State] -- A list of all of the states that can be directly reached from this state.
        """
        pass

    def get_predecessors(self) -> List[State] :
        """
        Returns a list of all states from which it is possible to transition to this state in one time step.

        Returns:
            List[State] -- A list of all states from which is it possible to transition directly to this state.
        """
        pass

    def get_transition_action(self, next_state : State) :
        """
        Returns a list of actions which can be taken in order to transition to a given next state.
        
        Arguments:
            next_state {State} -- The desired next state to find a transitional action for.

        Returns:
            List[Hashable] -- A list of actions which will allow the desired next state to be reached in one time step.
        """
        pass

    def generate_interaction_graph(self, initial_states : List[State]) :
        """
        Generates the state-transition graph for this environment.
        """

        states = []
        current_successor_states = []

        # Add initial states to current successor list.
        for initial_state in initial_states :
            current_successor_states.append(initial_state)

        # While we have no new successor states to process.
        while (not len(current_successor_states) == 0) :

            # Add each current new successor to our list of processed states.
            next_successor_states = []
            for successor_state in current_successor_states :
                if (not successor_state in current_successor_states) :
                    states.append(successor_state)

                    # Add this state's successors to the successor list.
                    if (not successor_state.is_terminal_state()) : 
                        for new_successor_state in successor_state.get_successors() :
                            next_successor_states.append(new_successor_state)

            current_successor_states = deepcopy(next_successor_states)
        
        # Create graph from state list.
        interaction_graph = nx.DiGraph()
        for state in states :
            
            # Add state to interaction graph.
            interaction_graph.add_node(state)

            for successor_state in state.get_successors() :
                interaction_graph.add_node(successor_state)
                interaction_graph.add_edge(state, successor_state)
        
        return interaction_graph
