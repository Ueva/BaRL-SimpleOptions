import math
import random
import copy

import numpy as np
import networkx as nx

from simpleoptions import BaseEnvironment, PrimitiveOption
from simpleoptions.implementations import SubgoalOptionGenerator, SubgoalOption

from typing import List, Hashable, Set, Dict

from tqdm import tqdm


class BetweennessOptionGenerator(SubgoalOptionGenerator):
    def __init__(
        self,
        initiation_set_size: int,
        option_learning_alpha: float,
        option_learning_epsilon: float,
        option_learning_gamma: float,
        option_learning_max_steps: int,
        option_learning_max_episode_steps: int,
        option_learning_default_action_value: float,
    ):
        super().__init__(
            option_learning_alpha,
            option_learning_epsilon,
            option_learning_gamma,
            option_learning_max_steps,
            option_learning_max_episode_steps,
            option_learning_default_action_value,
        )
        self.initiation_set_size = initiation_set_size

    def generate_options(
        self, env: BaseEnvironment, directed: bool, goal_states: List[Hashable] = None
    ) -> List["BetweennessOption"]:
        # Add primitive options to the environment.
        primitive_options = [PrimitiveOption(action, env) for action in env.get_action_space()]
        env.set_options(primitive_options)

        # Compute the betweenness centrality for each node in the STG.
        stg = env.generate_interaction_graph(directed=directed)
        if goal_states is None:
            centralities = nx.betweenness_centrality(stg, normalized=True, endpoints=True)
        else:
            centralities = nx.betweenness_centrality_subset(stg, sources=list(stg), targets=goal_states)

        # Find nodes that are local maxima of betweenness.
        local_maxima = [node for node in stg if self._is_local_maxima(node, stg, centralities)]

        # Define options for reaching each subgoal.
        options = [None for _ in range(len(local_maxima))]
        for i, subgoal in tqdm(enumerate(local_maxima), desc="Training Betweeness Options..."):
            initiation_set = sorted(list(nx.single_target_shortest_path_length(stg, subgoal)), key=lambda x: x[1])
            initiation_set = list(list(zip(*initiation_set))[0])[1 : self.initiation_set_size + 1]
            options[i] = BetweennessOption(
                env=env, subgoal=subgoal, initiation_set=initiation_set, betweenness=centralities[subgoal]
            )
            self.train_option(options[i])

        return options

    def _is_local_maxima(self, node: Hashable, stg: nx.Graph, centralities: Dict):
        return all(centralities[node] > centralities[neighbour] for neighbour in stg.neighbors(node))


class BetweennessOption(SubgoalOption):
    def __init__(
        self, env: BaseEnvironment, subgoal: Hashable, initiation_set: Set, betweenness: float, q_table: Dict = None
    ):
        super().__init__(env, subgoal, initiation_set, q_table)

        self.env = copy.copy(env)
        self.subgoal = subgoal
        self.initiation_set = initiation_set

        self.betweenness = betweenness

        if q_table is None:
            self.q_table = {}
        else:
            self.q_table = q_table

    def __str__(self):
        return f"BetweennessOption({self.subgoal})"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if isinstance(other, BetweennessOption):
            return self.subgoal == other.subgoal
        else:
            return False

    def __ne__(self, other):
        return not self == other


if __name__ == "__main__":
    from simpleenvs.envs.discrete_rooms import DiscreteXuFourRooms

    env = DiscreteXuFourRooms()
    gen = BetweennessOptionGenerator(20, 1.0, 0.2, 1.0, 10_000, 100, 0.0)
    options = gen.generate_options(env, directed=True)
    print([option.subgoal for option in options])

    option = options[0]
    print(option.subgoal)
    print(option.initiation_set)
    print(len(option.initiation_set))
    print(f"{option.initiation_set[1]}: {option.policy(option.initiation_set[1])}")
    print(
        {
            action: option.q_table[(hash(option.initiation_set[1]), hash(action))]
            for action in option.env.get_available_options(option.initiation_set[1])
        }
    )
