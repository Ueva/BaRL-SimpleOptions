from utils import create_graphs
import scipy.sparse
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

THETA = 1e-8

# DETERMINISTIC and directed version.#
####

# 1. Get the adjacency matrix M.
# 2. Compute the Directed laplacian.
# 3. Compute the right and left eigenvalues and eigenvectors of the laplacian.
# 4. Take the real parts of the eigenvalues and eigenvectors.
# 5. Compute Degree diagonal matrix from adjacency matrix.
# 6. Set temperature.


## Diffusion computation ##
# For every node:
# For every eigenvector:
#####


def train_option(self, env: BaseEnvironment, option: DiffusionOption):
    """
    Takes an Eigenoption and trains its internal policy using Value Iteration.

    Args:
        env (BaseEnvionrment): The environment in which to train the Eigenoption policies.
        option (Eigenoption): The option whose internal policy to train.
    """
    def _get_available_primitives(state) -> List:
        return env.get_available_actions(state)

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

    while True:
        # Policy Evaluation Step.
        while True:
            delta = 0
            for state in option.pvf.keys():
                if env.is_state_terminal(state):
                    continue

                v_old = values[state]

                action = policy[state]

                next_state = env.get_successors(state, [action])[0]
                reward = _intrinsic_reward(state, next_state)

                if env.is_state_terminal(next_state):
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

                next_state = env.get_successors(state, [action])[0]
                reward = _intrinsic_reward(state, next_state)

                if env.is_state_terminal(next_state):
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
    stg = env.generate_interaction_graph()
    for state in stg.nodes:
        if state in values:
            stg.nodes[state][f"PVF {option.pvf_id} Values"] = values[state]
        if state in policy:
            stg.nodes[state][f"PVF {option.pvf_id} Policy"] = str(
                policy[state])
    nx.write_gexf(stg, "eigen_test.gexf", prettyprint=True)

    option.primitive_policy = policy


class DiffusionOption(BaseOption):
    def __init__(self, env: BaseEnvironment, id: int):
        self.env = copy.copy(env)
        self.id = id
        self.primitive_policy = {}
        self.primitive_actions = {}
        for state in env.get_state_space():
            for action in env.get_available_actions(state):
                if action not in self.primitive_actions.keys():
                    self.primitive_actions[action] = PrimitiveOption(
                        action, env)

    def initiation(self, state):
        return not self.termination(state)

    def termination(self, state):
        return float(self.env.is_state_terminal(state))

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


if __name__ == "__main__":
    import numpy as np

    x = np.array([[
        2, -1, 0,
        -1, 2, -1,
        0, -1, 2
    ]]).reshape(3, 3)

    eigenvalues, eigenvectors = scipy.linalg.eigh(x)

    print(eigenvalues)


# env_name = 'maze_16_16.data'
env_name = 'four_rooms_11_11.data'
G, G2 = create_graphs(env_name)
G = G.to_undirected()
M = np.array(nx.adjacency_matrix(G).todense()).astype(
    np.double)  # adjacency_matrix
L = nx.directed_laplacian_matrix(nx.DiGraph(G))  # Laplacian
e, eigv_l, eigv_r = scipy.linalg.eig(
    L, left=True, right=True)  # Get eigenvectors of laplacian
o = e.argsort()  # Sort eigenvalues
e = e[o].real  # Make them real.
eigv_l = eigv_l[:, o].real.T  # eigenvectors left
eigv_r = eigv_r[:, o].real.T  # eigenvectors right
diags = M.sum(axis=1)  # Get the degree of each node
temperature = 8


def compute_diffusion(temp, e, eigv_l, eigv_r, diags):
    vals = []
    # For every node, for every
    for i in range(len(G.nodes)):
        val = 0
        for k in range(1, len(eigv_l)):
            val += np.power(1-0.5*e[k], temp) * np.power(diags[i], -0.5) * \
                eigv_l[k][i] * np.power(diags, 0.5)*eigv_r[k]
        val = np.power(np.linalg.norm(val), 2)
        vals.append(val)
    return vals


vals = compute_diffusion(temperature, e, eigv_l, eigv_r, diags)


pos = nx.get_node_attributes(G, 'pos')
nx.draw_networkx(G, pos=pos, node_color=vals, node_size=200, with_labels=True)
plt.show()


# stocastic example: modified M with an increased probability of moving up in the gridworld graph
G2 = G2.to_undirected()
M = np.array(nx.adjacency_matrix(G2).todense()).astype(np.double)

# Make the graph stocastic
for i in range(len(M)):
    for k in range(len(M)):
        if M[i][k] != 0 and i == k-1:
            M[i][k] += 4


rows_cols_to_delete = []
for i in G2.nodes:
    if i not in G.nodes:
        rows_cols_to_delete.append(i)

M = np.delete(M, rows_cols_to_delete, 0)
M = np.delete(M, rows_cols_to_delete, 1)

# Get the left and right eigenvectors of the laplacian
e, eigv_l, eigv_r = scipy.linalg.eig(L, left=True, right=True)

o = e.argsort()
e = e[o].real  # eigenvalues

eigv_l = eigv_l[:, o].real.T  # eigenvectors left
eigv_r = eigv_r[:, o].real.T  # eigenvectors right


diags = M.sum(axis=1)
temperature = 8

vals = compute_diffusion(temperature, e, eigv_l, eigv_r, diags)

nx.draw_networkx(G, pos=pos, node_color=vals, node_size=200, with_labels=True)
plt.show()
