import os
import math
import json
import random
import numpy as np
import networkx as nx

from option import Option
from option import PrimitiveOption
from option import SubgoalOption
from state import State

from abc import ABC, abstractmethod

class Environment(ABC) :
    """
    Abstract base class for environments, similar to OpenAI Gym environments.
    You should implement one-step dynamics for your environment, a "reset" function to
    initialise a the envionrment before starting an episode, and a function for returning
    the available options to the agent.

    It would be wise to make use of the methods that you have already defined in your implementation of
    the State class for your environment.
    """

    def __init__(self) :
        pass

    @abstractmethod
    def step(self, action) :
        """
        This method implements the one-step transition dynamics of the environment. Given an action,
        the environment transitions to some next state accordingly.
        
        Arguments:
            action {Hashable} -- The action for the agent to take in the current environmental state.

        Returns:
            State -- The next environmental state.
        """
        pass

    @abstractmethod
    def reset(self) :
        """
        This method initialises, or reinitialises, the environment prior to starting a new episode.
        It returns an initial state.

        Returns:
            State -- An initial environmental state.
        """
        pass

    @abstractmethod
    def get_available_options(self) :
        """
        This method returns the options (primitive options + subgoal options) which are available to the
        agent in the current environmental state.

        Returns:
            List[Option] -- The list of options available in this state.
        """
        pass


class BaseEnvironment(ABC) :
    """
    This abstract class implements a small amount of functionality which may be useful across many
    different environments, specifically to do with the enumeration of available options.
    
    Be sure to call the appropriate super-class methods when using this base class.
    """

    def __init__(self, options : List[Option]) :
        self.options = options
        self.current_state = None

    @abstractmethod
    def step(self) :
        pass

    @abstractmethod
    def reset(self) :
        pass

    def get_available_options(self) :
        # Loops through every option and sees whether the
        # current state is in its initiation set.
        available_options = []
        for option in self.options :
            if (option.initiation(self.current_state)) :
                available_options.append(option)
        
        return available_options
