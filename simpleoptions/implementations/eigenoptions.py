import math
import copy

import numpy as np
import scipy as sci
import networkx as nx

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption
from simpleoptions.implementations import GenericOptionGenerator

from tqdm import tqdm

TERMINATE_ACTION = "EIG_TERMINATE"
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

    def generate_options(self, env: BaseEnvironment, return_pvfs: bool = False):
        """
        Generates a set of Eigenoptions for the given environment.

        Args:
            env (BaseEnvironment): _description_
            return_pvfs (bool, optional): _description_. Defaults to False.

        Returns:
            dict[Eigenoption]: A dictionary of the generated Eigenoptions, keyed by Eigenoption ID.
            dict[dict[str]] (optional): A dictionary containing each Eigenoption's PVF, mapping states to proto-values, keyed by Eigenoption ID.

        """
        env.reset()

        if hasattr(env, "get_successor_representation"):
            eigenoptions, pvfs = self._generate_from_sr(env)
        else:
            eigenoptions, pvfs = self._generate_from_laplacian(env)

        if return_pvfs:
            return eigenoptions, pvfs
        else:
            return eigenoptions

    def _generate_from_laplacian(self, env: BaseEnvironment):
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

        eigenoptions = {pvf_id: Eigenoption(env, pvf, pvf_id) for pvf_id, pvf in pvfs.items()}

        for eigenoption in tqdm(eigenoptions.values(), desc="Training Eigenoptions..."):
            self.train_option(env, eigenoption)

        return eigenoptions, pvfs

    def _generate_from_sr(self, env: BaseEnvironment):
        # TODO: Implement generating Eigenoptions from the Successor Representation.
        eigenoptions, pvfs = self._generate_from_laplacian(env)
        return eigenoptions, pvfs

    def train_option(self, env: BaseEnvironment, option: "Eigenoption"):
        """
        Takes an Eigenoption and trains its internal policy using Value Iteration.

        Args:
            env (BaseEnvionrment): The environment in which to train the Eigenoption policies.
            option (Eigenoption): The option whose internal policy to train.
        """

        def _get_available_primitives(state):
            return ["EIG_TERMINATE"] + [action for action in env.get_available_actions(state)]

        def _intrinsic_reward(state, next_state):
            reward = option.pvf[next_state] - option.pvf[state]

            if math.isclose(reward, 0, abs_tol=1e-6):
                return 0
            else:
                return reward

        env.reset()

        policy = {}
        values = {}

        # Initialise values and policy.
        for state in option.pvf.keys():
            values[state] = 0

            if not env.is_state_terminal(state):
                policy[state] = TERMINATE_ACTION

        while True:
            # Policy Evaluation Step.
            while True:
                delta = 0
                for state in option.pvf.keys():
                    if env.is_state_terminal(state):
                        continue

                    v_old = values[state]

                    action = policy[state]
                    if action == TERMINATE_ACTION:
                        next_state = TERMINATE_ACTION
                        reward = 0
                    else:
                        next_state = env.get_successors(state, [action])[0]
                        reward = _intrinsic_reward(state, next_state)

                    if next_state == TERMINATE_ACTION or env.is_state_terminal(next_state):
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
                if env.is_state_terminal(state):
                    continue

                a_old = policy[state]

                best_action = None
                best_value = -np.inf
                for action in _get_available_primitives(state):
                    if action == TERMINATE_ACTION:
                        next_state = TERMINATE_ACTION
                        reward = 0
                    else:
                        next_state = env.get_successors(state, [action])[0]
                        reward = _intrinsic_reward(state, next_state)

                    if next_state == TERMINATE_ACTION:
                        v_next = 0
                    elif env.is_state_terminal(next_state):
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

        # Add policy to graph for inspection.
        # stg = env.generate_interaction_graph()
        # for state in stg.nodes:
        #     if state in values:
        #         stg.nodes[state][f"PVF {option.pvf_id} Values"] = values[state]
        #     if state in policy:
        #         stg.nodes[state][f"PVF {option.pvf_id} Policy"] = str(policy[state])
        # nx.write_gexf(stg, "eigen_test.gexf", prettyprint=True)

        option.primitive_policy = policy


class Eigenoption(BaseOption):
    def __init__(self, env: BaseEnvironment, pvf: dict, pvf_id: str):
        self.env = copy.copy(env)
        self.pvf = pvf
        self.pvf_id = pvf_id
        self.primitive_policy = {}

        self.primitive_actions = {}
        for state in env.get_state_space():
            for action in env.get_available_actions(state):
                if action not in self.primitive_actions.keys():
                    self.primitive_actions[action] = PrimitiveOption(action, env)

    def initiation(self, state):
        return not self.termination(state)

    def termination(self, state):
        if self.primitive_policy[state] == TERMINATE_ACTION or self.env.is_state_terminal(state):
            return float(True)
        else:
            return float(False)

    def policy(self, state, test=False):
        return self.primitive_actions[self.primitive_policy[state]]

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


# if __name__ == "__main__":
#     from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

#     env = DiscreteXuFourRooms()
#     gen = EigenoptionGenerator(4, 0.99)
#     options = gen.generate_options(env)

#     gen.train_option(env, options["1"])
