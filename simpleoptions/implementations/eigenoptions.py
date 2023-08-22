import os
import math
import copy

import numpy as np
import scipy as sci
import networkx as nx

from tqdm import tqdm
from typing import List, Dict, Set

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption
from simpleoptions.implementations import GenericOptionGenerator


TERMINATE_ACTION = "EIG_TERMINATE"
TERMINATE_STATE = "EIG_TERMINAL"
THETA = 1e-8


class EigenoptionGenerator(GenericOptionGenerator):
    def __init__(self, num_pvfs: int, gamma: float):
        """
        Instantiates a new EigenOption Generator.

        Args:
            num_pvfs (int): The number of Proto-Value Functions (PVFs) to derive Eigenoptions from.
                            The number of Eigenoptions generated will be double this number, since both directions of each PVF will be considered.
            gamma (float): The discount factor to use when training the option policies using value iteration.
        """
        self.num_pvfs = num_pvfs
        self.gamma = gamma

    def generate_options(
        self, env: BaseEnvironment, return_pvfs: bool = False, debug: bool = False
    ) -> List["Eigenoption"]:
        """
        Generates a set of Eigenoptions for the given environment.

        Args:
            env (BaseEnvironment): The environment to generate eigenoptions in.
            return_pvfs (bool, optional): Whether to return the list of PVF. Defaults to False.
            debug (bool, options):

        Returns:
            list[Eigenoption]: A list containing the generated Eigenoptions.
            dict[dict[str]] (optional): A dictionary containing each Eigenoption's PVF, mapping states to proto-values, keyed by Eigenoption pvf_id.

        """
        env.reset()

        if hasattr(env, "get_successor_representation"):
            eigenoptions, pvfs = self._generate_from_sr(env, debug)
        else:
            eigenoptions, pvfs = self._generate_from_laplacian(env, debug)

        if return_pvfs:
            return eigenoptions, pvfs
        else:
            return eigenoptions

    def _generate_from_laplacian(self, env: BaseEnvironment, debug: bool = False):
        # Generate the environment's state-transition graph and ensure that it is undirected.
        stg = env.generate_interaction_graph()
        if isinstance(stg, nx.DiGraph):
            stg = stg.to_undirected()

        node_list = list(stg.nodes())

        # Compute the normalised graph Laplacian.
        laplacian = nx.normalized_laplacian_matrix(stg, nodelist=node_list)

        # Compute the eigenvectors and eigenvalues of the graph laplacian.
        vals, vecs = sci.linalg.eigh(laplacian.todense())

        # Create dictionary mapping states to proto-value functions.
        pvfs = {}
        for i in range(min(self.num_pvfs, len(vecs))):
            pvfs[f"{i}"] = {}
            pvfs[f"-{i}"] = {}
            pvf = vecs[:, i]

            for j, node in enumerate(node_list):
                pvfs[f"{i}"][node] = pvf[j]
                pvfs[f"-{i}"][node] = -pvf[j]

        eigenoptions = [Eigenoption(env, pvf, pvf_id) for pvf_id, pvf in pvfs.items()]

        for i, eigenoption in tqdm(enumerate(eigenoptions), desc="Training Eigenoptions..."):
            self.train_option(eigenoptions[i])

        # If Debugging, output annotated graph for inspection.
        if debug:
            stg = env.generate_interaction_graph()
            for eigenoption in eigenoptions:
                for state in stg.nodes:
                    if state in eigenoption.state_values:
                        stg.nodes[state][f"PVF {eigenoption.pvf_id} Values"] = eigenoption.state_values[state]
                        print(f"PVF {eigenoption.pvf_id} Values")
                    if state in eigenoption.primitive_policy:
                        stg.nodes[state][f"PVF {eigenoption.pvf_id} Policy"] = str(eigenoption.policy(state))
            nx.write_gexf(stg, "eigen_test.gexf", prettyprint=True)

        return eigenoptions, pvfs

    def _generate_from_sr(self, env: BaseEnvironment, debug: bool = False):
        # TODO: Implement generating Eigenoptions from the Successor Representation.
        eigenoptions, pvfs = self._generate_from_laplacian(env, debug)
        return eigenoptions, pvfs

    def train_option(self, option: "Eigenoption"):
        """
        Takes an Eigenoption and trains its internal policy using Value Iteration.

        Args:
            option (Eigenoption): The option whose internal policy to train.
        """

        def _get_available_primitives(state):
            return ["EIG_TERMINATE"] + [action for action in option.env.get_available_actions(state)]

        def _intrinsic_reward(state, next_state):
            reward = option.pvf[next_state] - option.pvf[state]

            if math.isclose(reward, 0, abs_tol=1e-6):
                return 0
            else:
                return reward

        option.env.reset()

        policy = {}
        values = {}

        # Initialise values and policy.
        for state in option.pvf.keys():
            values[state] = 0
            policy[state] = TERMINATE_ACTION

        while True:
            # Policy Evaluation Step.
            while True:
                delta = 0
                for state in option.pvf.keys():
                    if option.env.is_state_terminal(state):
                        continue

                    v_old = values[state]

                    action = policy[state]
                    if action == TERMINATE_ACTION:
                        next_state = TERMINATE_STATE
                        reward = 0
                    else:
                        next_state = option.env.get_successors(state, [action])[0]
                        reward = _intrinsic_reward(state, next_state)

                    if next_state == TERMINATE_STATE or option.env.is_state_terminal(next_state):
                        v_next = 0
                    else:
                        v_next = values[next_state]

                    values[state] = reward + self.gamma * v_next

                    delta = max(delta, abs(v_old - values[state]))
                if delta < THETA:
                    break

            # Policy Improvement Step.
            policy_stable = True
            for state in option.pvf.keys():
                if option.env.is_state_terminal(state):
                    continue

                a_old = policy[state]

                best_action = None
                best_value = -np.inf
                for action in _get_available_primitives(state):
                    if action == TERMINATE_ACTION:
                        next_state = TERMINATE_STATE
                        reward = 0
                    else:
                        next_state = option.env.get_successors(state, [action])[0]
                        reward = _intrinsic_reward(state, next_state)

                    if next_state == TERMINATE_STATE:
                        v_next = 0
                    elif option.env.is_state_terminal(next_state):
                        continue
                    else:
                        v_next = values[next_state]

                    value = reward + self.gamma * v_next

                    if value > best_value:
                        best_action = action
                        best_value = value

                policy[state] = best_action

                if a_old != policy[state]:
                    policy_stable = False

            if policy_stable:
                break

        option.primitive_policy = policy
        option.state_values = values


class Eigenoption(BaseOption):
    def __init__(self, env: BaseEnvironment, pvf: dict, pvf_id: str):
        self.env = copy.copy(env)
        self.pvf = pvf
        self.pvf_id = pvf_id
        self.primitive_policy = {}
        self.state_values = {}

        # Add primitive options to the environment.
        primitive_options = [PrimitiveOption(action, self.env) for action in self.env.get_action_space()]
        self.env.set_options(primitive_options)

        self.primitive_actions = {
            option.action: option for option in self.env.options if isinstance(option, PrimitiveOption)
        }

    def initiation(self, state):
        return not self.termination(state)

    def termination(self, state):
        if self.primitive_policy[state] == TERMINATE_ACTION or self.env.is_state_terminal(state):
            return float(True)
        else:
            return float(False)

    def policy(self, state, test=False):
        action = self.primitive_policy[state]
        if action == TERMINATE_ACTION:
            return TERMINATE_ACTION
        else:
            return self.primitive_actions[action]

    def set_primitive_policy(self, primitive_policy: dict):
        self.primitive_policy = primitive_policy

    def __str__(self):
        return f"Eigenoption({self.pvf_id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, Eigenoption):
            return self.pvf_id == other.pvf_id
        else:
            return False

    def __ne__(self, other):
        return not self == other


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

    env = DiscreteXuFourRooms()
    gen = EigenoptionGenerator(5, 0.9)
    options = gen.generate_options(env, debug=True)
