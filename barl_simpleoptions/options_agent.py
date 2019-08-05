import os
import math
import random
import numpy as np
import networkx as nx

from copy import deepcopy
from typing import List

from barl_simpleoptions.option import Option
from barl_simpleoptions.state import State
from barl_simpleoptions.environment import Environment

class OptionAgent :
    """
    An agent which acts in a given environment, learning using the Macro-Q learning
    and intra-option learning algorithms.
    """

    def __init__(self, env : Environment, epsilon : float, alpha : float, gamma : float) :
        """
        Constructs a new OptionAgent object.

        Arguments:
            env {Environment} -- The environment for the agent to act in.
            epsilon {float} -- The chance of the agent taking a random action when following its base policy.
            alpha {float} -- The agent's learning rate.
            gamma {float} -- The environment's decay factor.
        """

        self.q_table = {}
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.current_option = None
        self.current_option_initiation_state = None

    def macro_learn(self, initiation_state : State, option : Option, rewards : List[float], termination_state : State) :
        """
        Performs a macro-Q learning update.
        
        Arguments:
            initiation_state {State} -- The state in which this option was initiated.
            option {Option} -- The option which was executed.
            rewards {List[float]} -- The rewards earned at each time step while the option was executing.
            termination_state {State} -- The state in which the option terminated.
        """
        
        # Perform macro-q learning update.
        old_value = self.q_table.get((str(initiation_state), str(option)), 0)
    
        discounted_sum_of_rewards = 0
        for i in range (0, len(rewards)) :
            discounted_sum_of_rewards += math.pow(self.gamma, i) * rewards[0]

        # Get Q-Values for Next State.
        q_values = [self.q_table.get((str(termination_state), str(o)), 0) for o in self.env.get_available_options(termination_state)]
        
        # Cater for terminal states.
        if (len(q_values) == 0) :
            q_values.append(0)
        
        max_q = max(q_values)

        self.q_table[(str(initiation_state), str(option))] = old_value + self.alpha * (math.pow(self.gamma, len(rewards)) * max_q - old_value + discounted_sum_of_rewards)
        

    def intra_option_learn(self, state : State, action, reward, next_state : State) :
        """
        Performs a one-step intra-option learning update.
        
        Arguments:
            state {State} -- The state in which the primitive action was executed.
            action {[type]} -- The executed primitive action.
            reward {[type]} -- The reward earned for executing the primitive action.
            next_state {State} -- The state reached after executing the primitive action.
        """

        # Perform one-step intra-option learning update.

        # For each option, if the action just taken is the same action
        # as that option would have specified, then perform a one-step
        # intra-option q-learning update.
        for option in self.env.get_available_options(state) :
            if (option.initiation(state) and option.policy(state) == action) :

                # Get Q-Values for Next State.
                q_values = [self.q_table.get((str(next_state), str(o)), 0) for o in self.env.get_available_options(next_state)]

                # Cater for Terminal States.
                if (len(q_values) == 0) :
                    q_values.append(0)

                max_q = max(q_values)

                value = reward + self.gamma * max_q
                
                old_value = self.q_table.get((str(state), str(option)), 0)

                self.q_table[(str(state), str(option))] = old_value + self.alpha * (value - old_value)

    def select_action(self, state : State) -> Option :
        """
        Returns the selected action for the given state.
        
        Arguments:
            state {State} -- The state in which to select an action.
        
        Returns:
            {hashable} -- The identifier of the selected action.
        """

        # Select option from set of available options
        # Use epsilon greedy at top level, use option policy at option level.

        # If we are not currently following an option policy, act according
        # to the epsilon-greedy policy over the set of currently available options.
        if (self.current_option is None) :
            available_options = self.env.get_available_options(state)

            # Random Action.
            if (random.random() < self.epsilon) :
                return random.choice(available_options)

            # Best Action.
            else :
                q_values = [self.q_table.get((str(state), str(o)), 0) for o in available_options]
                max_q = max(q_values)

                return available_options[q_values.index(max_q)]
        
        # If we are currently following an option policy, return it.
        else :
            return self.current_option

    def run_agent(self, num_episodes : int) :
        """
        Runs the agent for a given number of episodes.
        
        Arguments:
            num_episodes {int} -- The number of episodes to run the agent for.

        Returns:
            {List[float]} -- A list containing the reward earned during each episode.
        """

        episode_rewards = []

        for episode_i in range(0, num_episodes) :
            
            # Initialise initial state.
            state = self.env.reset()
            sum_rewards = 0 
            terminal = False
            option_rewards = []

            while (not terminal) :

                # Select action from root policy.
                if (self.current_option is None) :
                    option = self.select_action(state)
                    self.current_option = option
                    self.current_option_initiation_state = deepcopy(state)
                    action = option.policy(state)
                # Select action from option policy.
                else :
                    option = self.select_action(state)
                    action = option.policy(action)

                # Take action, observe reward, next state, terminal.
                next_state, reward, terminal = env.step(action)
                option_rewards.append(reward)

                # Perform one-step intra-option learning update.
                self.intra_option_learn(state, action, reward, next_state)

                # If we have left the initiation set of the currently executing option, terminate
                # the option and perform a macro-Q learning update.
                if (self.current_option.termination(next_state) or terminal) :
                    self.macro_learn(self.current_option_initiation_state, self.current_option, option_rewards, next_state)
                    option_rewards = []
                    self.current_option = None
                    self.current_option_initiation_state = None
                
                # Update cumulative episode rewards.
                sum_rewards += reward

                # Update current state.
                state = next_state

            # Record the cumulative rewards earned during thsi episode.
            episode_rewards.append(sum_rewards)
        
        return episode_rewards