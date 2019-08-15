import os
import math
import json
import random
import numpy as np
import networkx as nx

from abc import ABC, abstractmethod

from barl_simpleoptions.state import State

class Option(ABC) :
    """
    Interface for a reinforcement learning option.
    """

    def __init__(self) :
        pass

    @abstractmethod
    def initiation(self, state : 'State') -> bool :
        """
        Returns whether or not a given state is in this option's initation set.
        
        Arguments:
            state {State} -- The state whose membership to the initiation set is
            to be tested.
            
            Returns:
                bool -- [description]
        """
        pass

    @abstractmethod
    def policy(self, state : 'State') :
        """
        Returns the action specified by this option's policy for a given state.
        
        Arguments:
            state {State} -- The environmental state in which the option chooses an action in.

        Returns:
            action [Hashable] -- The action specified by the option's policy in this state.
        """
        pass

    @abstractmethod
    def termination(self, state : 'State') -> bool :
        """
        Returns whether or not the option terminates in the given state.
        
        Arguments:
            state {State} -- The state in which to test for termination.
        
        Returns:
            bool -- Whether or not this option terminates in this state.
        """
        pass


class PrimitiveOption(Option) :
    """
    Class representing a primitive option.
    Primitive options terminate with probability one in every state, and have
    an initiation set consisting of all of the states where their underlying
    primitive actions are available.
    """

    def __init__(self, action) :
        """Constructs a new primitive option.
        
        Arguments:
            action {Hashable} -- The underlying primitive action for this option.
        """
        self.action = action

    def initiation(self, state : State) :
        return state.is_action_legal(self.action)

    def policy(self, state : State) :
        return self.action

    def termination(self, state : State) :
        return True

    def __str__(self) :
        return "PrimitiveOption({})".format(self.action)

    def __repr__(self) :
        return str(self)


class SubgoalOption(Option) :
    """
    Class representing a temporally abstract action in reinforcement learning through a policy to
    be executed between an initiation state and a termination state.
    """
    
    def __init__(self, subgoal : 'State', graph : nx.DiGraph, policy_file_path : str, initiation_set_size : int) :
        """Constructs a new subgoal option.
        
        Arguments:
            subgoal {State} -- The state to act as this option's subgoal.
            graph {nx.Graph} -- The state interaction graph of the reinforcement learning environment.
            policy_file_path {str} -- The path to the file containing the policy for this option.
            initiation_set_size {int} -- The size of this option's initiation set.
        """
        
        self.graph = graph
        self.subgoal = subgoal
        self.policy_file_path = policy_file_path
        self.initiation_set_size = initiation_set_size

        self._build_initiation_set()

        # Load the policy file for this option.
        with open(policy_file_path, mode = "rb") as f:
            self.policy_dict = json.load(f)

    def initiation(self, state : 'State') :
        return str(state) in self.initiation_set

    def policy(self, state : 'State') :
        return self.policy_dict[str(state)]

    def termination(self, state : State) :
        return (state == self.subgoal) or (not self.initiation(state))

    def _build_initiation_set(self) :
        """
        Constructs the intiation set for this subgoal option.
        The initation set consists of the initiation_set_size closest states to the subgoal state.
        """
        
        # Get distances of all nodes from subgoal node.
        node_distances = []
        for node in self.graph :
            # Exclude subgoal node.
            if (not node == str(self.subgoal)) :
                # Only consider nodes which can reach the subgoal node.
                if (nx.has_path(self.graph, source = str(node), target = str(self.subgoal))) :
                    # Append the tuple (node, distance from subgoal) to the list.
                    node_distances.append((node, len(list(nx.shortest_path(self.graph, source = node, target = str(self.subgoal))))))
        
        # Sort the list of nodes by distance from the subgoal.
        node_distances = sorted(node_distances, key = lambda x: x[1])
        
        # Take the closest set_size nodes to the subgoal as the initiation set.
        initiation_set, _ = zip(*node_distances)
        if (len(initiation_set) > self.initiation_set_size) :
            self.initiation_set = list(initiation_set)[:self.initiation_set_size].copy()
        else :
            self.initiation_set = list(initiation_set).copy()

    def __str__(self) :
        return "SubgoalOption({}~{})".format(str(self.subgoal, self.policy_file_path))

    def __repr__(self):
        return str(self)
