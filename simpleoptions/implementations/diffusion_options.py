import scipy as sp
import sys
import matplotlib.pyplot as plt
import pickle
import math
import copy

import numpy as np
import scipy
import networkx as nx

from simpleoptions import BaseEnvironment, BaseOption, PrimitiveOption
from simpleoptions.implementations import GenericOptionGenerator
from typing import List
from tqdm import tqdm


class DiffusionOption(BaseOption):
    def __init__(self, env: BaseEnvironment, id: int):
        self.env = copy.copy(env)
        self.id = id
        self.primitive_policy = {}
        self.primitive_actions = {}
        for state in env.get_state_space():
            for action in env.get_available_actions(state):
                if action not in self.primitive_actions.keys():
                    self.primitive_actions[action] = PrimitiveOption(action, env)

    def initiation(self, state):
        return not self.termination(state)

    def termination(self, state):
        return self.env.is_state_terminal(state) or state == self.subgoal_state

    def policy(self, state, test=False):
        return self.primitive_actions[self.primitive_policy[state]]

    def set_primitive_policy(self, primitive_policy: dict):
        self.primitive_policy = primitive_policy

    def __str__(self):
        return f"DiffusionOption({self.id})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, DiffusionOption):
            return self.id == other.id
        else:
            return False

    def __ne__(self, other):
        return not self == other


class DiffusitionOptionGenerator(GenericOptionGenerator):
    def __init__(self, num_options: int, time_scale: int):
        self.num_options = num_options
        self.time_scale = time_scale

    def generate_options(self, env: BaseEnvironment) -> None:
        """
        Generates a set of Diffusion Options for the given environment.

        Args:
            env (BaseEnvironment): Environment used to generate Diffusion Options.

        Returns:
            List: A list of the generated Diffusion options.
        """
        env.reset()
        diffusion_options = self._generate_options(env)
        return diffusion_options

    def _generate_options(self, env: BaseEnvironment):
        # Generate the environment's state-transition graph and ensure that it is undirected.
        stg, node_list, adj_mat, deg_mat, empty_rc = self._extract_graph_matrices(env)
        e_vals, left_e_vecs, right_e_vecs, D_root = self._compute_eigendecomp(deg_mat, adj_mat)
        f_vec = self.compute_f(e_vals, left_e_vecs, right_e_vecs, D_root)

        f_idx = 0
        f_dict = {}
        for i, node in enumerate(node_list):
            if i in empty_rc:
                f_dict[node] = 0
            else:
                f_dict[node] = f_vec[f_idx]
                f_idx += 1

        print(f_dict)
        nx.set_node_attributes(stg, values=f_dict, name="diffusion_score")
        subgoals = self._detect_subgoals_from_graph(stg)
        options = [DiffusionOption(env, subgoal) for subgoal in subgoals]
        return options

    def _compute_eigendecomp(self, D, adj_mat):
        D_diag = np.diag(D)
        D_root = np.diag(np.sqrt(D_diag))
        D_root_inv = np.diag(1 / np.sqrt(D_diag))

        N = D_root_inv @ (D - adj_mat) @ D_root_inv

        _, R = scipy.linalg.polar(N, side="left")
        e_vals, left_e_vecs, right_e_vecs = scipy.linalg.eig(R, left=True, right=True)

        e_order = np.argsort(e_vals)
        e_vals = np.real(e_vals[e_order])
        left_e_vecs = np.real(left_e_vecs[:, e_order]).T
        right_e_vecs = np.real(right_e_vecs[:, e_order]).T

        return e_vals, left_e_vecs, right_e_vecs, D_root

    def _extract_graph_matrices(self, env: BaseEnvironment):
        stg = env.generate_interaction_graph()
        undirected_stg = stg.to_undirected()
        node_list = list(undirected_stg.nodes)

        adj_mat = nx.adjacency_matrix(stg, nodelist=node_list).todense()

        # Remove unused nodes.
        empty_r = np.flatnonzero(adj_mat.sum(1) == 0)
        empty_c = np.flatnonzero(adj_mat.sum(1) == 0)
        empty_rc = np.union1d(empty_r, empty_c)
        adj_mat = np.delete(adj_mat, empty_rc, axis=0)
        adj_mat = np.delete(adj_mat, empty_rc, axis=1)

        # Degree matrix
        deg_mat = np.diag(np.sum(adj_mat, axis=1))

        return undirected_stg, node_list, adj_mat, deg_mat, empty_rc

    def compute_f(self, e_vals, left_e_vecs, right_e_vecs, D_root):
        e_vals = 1 - 0.5 * e_vals
        e_val_pow = e_vals**self.time_scale
        d_root_vec = np.diag(D_root)
        f = np.zeros(right_e_vecs.shape[0])

        for s in range(f.shape[0]):
            f_vec = 0
            for i in range(1, len(e_vals)):
                f_vec += e_val_pow[i] * left_e_vecs[i, s] * right_e_vecs[i]
            f_vec /= d_root_vec[s]
            f_vec *= d_root_vec
            f[s] = np.linalg.norm(f_vec, ord=2) ** 2
        return f

    def _add_f_score_to_graph(self, stg, f_dict):
        for node in f_dict:
            stg.nodes[node]["diffusion_score"] = f_dict[node]
        return stg

    def _detect_subgoals_from_graph(self, stg):
        subgoals = {}

        for node in list(stg.nodes()):
            is_local_maxima = True
            neighbours = list(stg.neighbors(node))
            if len(neighbours) == 0:
                continue

            for neighbour in list(neighbours):  # N.B. Only considers out_edges if stg is directed
                if stg.nodes[neighbour]["diffusion_score"] > stg.nodes[node]["diffusion_score"]:
                    is_local_maxima = False
                    break

            if is_local_maxima:
                subgoals[node] = stg.nodes[node]["diffusion_score"]

                # Only take the top K subgoals.
                sorted_subgoals = dict(sorted(subgoals.items(), key=lambda x: x[1], reverse=True)[: self.num_options])
                return list(sorted_subgoals.keys())

    def train_option(self, env: BaseEnvironment, option: DiffusionOption):
        """
        Takes an Eigenoption and trains its internal policy using Value Iteration.

        Args:
            env (BaseEnvionrment): The environment in which to train the Eigenoption policies.
            option (Eigenoption): The option whose internal policy to train.
        """

        def _get_available_primitives(state) -> List:
            return env.get_available_actions(state)

        def _pseudo_reward(next_state, option):
            if next_state == option.subgoal:
                return 1
            else:
                return -0.01

        pass


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

    env = DiscreteXuFourRooms()

    option_generator = DiffusitionOptionGenerator(10, 8)
    print(len(env.get_state_space()))
    options = option_generator.generate_options(env)

    print(options)
