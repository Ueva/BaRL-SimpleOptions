import copy

import numpy as np
import scipy
import networkx as nx

from simpleoptions import BaseEnvironment, PrimitiveOption
from simpleoptions.implementations import SubgoalOptionGenerator, SubgoalOption

from typing import List, Set, Dict, Hashable

from tqdm import tqdm


class DiffusionOptionGenerator(SubgoalOptionGenerator):
    def __init__(
        self,
        num_options: int,
        time_scale: int,
        option_learning_alpha: float,
        option_learning_epsilon: float,
        option_learning_gamma: float,
        option_learning_max_steps: int,
        option_learning_max_episode_steps: int,
        option_learning_default_action_value: float,
        *args,
        **kwargs,
    ):
        super().__init__(
            option_learning_alpha,
            option_learning_epsilon,
            option_learning_gamma,
            option_learning_max_steps,
            option_learning_max_episode_steps,
            option_learning_default_action_value,
        )
        self.num_options = num_options
        self.time_scale = time_scale

    def generate_options(
        self, env: BaseEnvironment, return_subgoals: bool = False, debug: bool = False
    ) -> List["DiffusionOption"]:
        """
        Generates a set of Diffusion Options for the given environment.

        Args:
            env (BaseEnvironment): Environment used to generate Diffusion Options.

        Returns:
            List: A list of the generated Diffusion options.
        """
        # Add primitive options to the environment.
        primitive_options = [PrimitiveOption(action, env) for action in env.get_action_space()]
        env.set_options(primitive_options)
        env.reset()

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

        nx.set_node_attributes(stg, values=f_dict, name="diffusion_score")

        # Select all local maxima of diffusion score as subgoals.
        subgoals = [node for node in stg if self._is_local_maxima(node, stg, f_dict)]
        subgoals = sorted(subgoals, key=lambda x: f_dict[x], reverse=True)[: min(self.num_options, len(subgoals))]

        # Create and train an option for reaching each subgoal.
        options = [None for _ in range(len(subgoals))]
        for i, subgoal in tqdm(enumerate(subgoals), desc="Training Diffusion Options..."):
            # Set initiation set to be all non-terminal states from which there exists a path to the subgoal state.
            initiation_set = set(
                [
                    state
                    for state in env.get_state_space()
                    if (not env.is_state_terminal(state)) and (nx.has_path(stg, state, subgoal))
                ]
            )

            options[i] = DiffusionOption(env, subgoal, initiation_set - {subgoal})
            self.train_option(options[i])

        # If Debugging, output annotated graph for inspection.
        if debug:
            stg = env.generate_interaction_graph()
            for option in options:
                for state in stg.nodes:
                    if option.initiation(state):
                        stg.nodes[state][f"{option.subgoal} Policy"] = str(option.policy(state))
                    elif state == option.subgoal:
                        stg.nodes[state][f"{option.subgoal} Policy"] = "GOAL"
            nx.write_gexf(stg, "diffusion_test.gexf", prettyprint=True)

        if return_subgoals:
            return options, subgoals
        else:
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

    def _is_local_maxima(self, node: Hashable, stg: nx.Graph, centralities: Dict):
        return all(
            centralities[node] > centralities[neighbour] for neighbour in stg.neighbors(node) if neighbour != node
        )


class DiffusionOption(SubgoalOption):
    def __init__(self, env: BaseEnvironment, subgoal: Hashable, initiation_set: Set[Hashable], q_table: Dict = None):
        super().__init__(env, subgoal, initiation_set, q_table)

    def __str__(self):
        return f"DiffusionOption({self.subgoal})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, DiffusionOption):
            return self.subgoal == other.subgoal
        else:
            return False

    def __ne__(self, other):
        return not self == other


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

    env = DiscreteXuFourRooms()

    option_generator = DiffusionOptionGenerator(10, 8, 1.0, 0.3, 1.0, 100_000, 500, 0.0)
    options = option_generator.generate_options(env, debug=True)
