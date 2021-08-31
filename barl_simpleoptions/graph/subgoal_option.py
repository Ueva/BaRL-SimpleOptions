import json
import networkx as nx

from typing import Hashable

from barl_simpleoptions.option import Option


class SubgoalOption(Option):
    """
    Class representing a temporally abstract action in reinforcement learning through a policy to
    be executed between an initiation state and a termination state.
    """

    def __init__(self, subgoal: Hashable, graph: nx.DiGraph, policy_file_path: str, initiation_set_size: int):
        """
        Constructs a new subgoal option.

        Arguments:
            subgoal {Hashable} -- The state to act as this option's subgoal.
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
        with open(policy_file_path, mode="rb") as f:
            self.policy_dict = json.load(f)

    def initiation(self, state: Hashable):
        return hash(state) in self.initiation_set

    def policy(self, state: Hashable):
        return self.policy_dict[hash(state)]

    def termination(self, state: Hashable):
        return (state == self.subgoal) or (not self.initiation(state))

    def _build_initiation_set(self):
        """
        Constructs the intiation set for this subgoal option.
        The initation set consists of the initiation_set_size closest states to the subgoal state.
        """

        # Get distances of all nodes from subgoal node.
        node_distances = []
        for node in self.graph:
            # Exclude subgoal node.
            if not node == hash(self.subgoal):
                # Only consider nodes which can reach the subgoal node.
                if nx.has_path(self.graph, source=hash(node), target=hash(self.subgoal)):
                    # Append the tuple (node, distance from subgoal) to the list.
                    node_distances.append(
                        (node, len(list(nx.shortest_path(self.graph, source=node, target=hash(self.subgoal)))))
                    )

        # Sort the list of nodes by distance from the subgoal.
        node_distances = sorted(node_distances, key=lambda x: x[1])

        # Take the closest set_size nodes to the subgoal as the initiation set.
        initiation_set, _ = zip(*node_distances)
        if len(initiation_set) > self.initiation_set_size:
            self.initiation_set = list(initiation_set)[: self.initiation_set_size].copy()
        else:
            self.initiation_set = list(initiation_set).copy()

    def __str__(self):
        return "SubgoalOption({}~Nearest{})".format(str(self.subgoal), str(self.initiation_set_size))

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return str(self)