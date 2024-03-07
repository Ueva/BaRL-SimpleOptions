import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simpleoptions.function_approximation import ApproxBaseEnvironment
from simpleoptions.function_approximation.utils import build_simple_network
from simpleoptions.function_approximation.agents.replay_buffer import HierarchicalReplayBuffer

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, DefaultDict


class DeepOptionsAgent:
    def __init__(
        self,
        env: ApproxBaseEnvironment,
        test_env: ApproxBaseEnvironment,
        network: Dict = None,
        memory_capacity: int = 1e6,
    ):
        self.env = env
        self.test_env = test_env

        # Build online q-network for action selection.
        self.online = deepcopy(network)

        # Build the target q-network for computing td targets.
        self.target = deepcopy(network)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def run_agent(
        self,
        num_epochs: int,
        epoch_length: int,
        render_interval: int = 0,
        test_interval: int = 0,
        test_length: int = 0,
        test_runs: int = 0,
        verbose_logging: bool = True,
        episodic_eval: bool = False,
    ) -> Tuple[DefaultDict, DefaultDict | None]:
        pass


if __name__ == "__main__":
    env = gym.make("Acrobot-v1")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    cnn_net_arch = {
        "observation_dim": 3,
        "action_dim": 4,
        "activation": nn.PReLU,
        "input": {
            "type": "conv2d",
            "out": 32,
            "kernel_size": 8,
            "stride": 4,
        },
        "output": {
            "in": 128,
        },
        "cnn": {
            "num": 2,
            "ins": [32, 64],
            "outs": [64, 64],
            "kernel_sizes": [4, 3],
            "strides": [4, 1],
        },
        "linear": {
            "num": 2,
            "ins": [2048, 512],
            "outs": [512, 128],
        },
    }

    fc_net_arch = {
        "observation_dim": obs_dim,
        "action_dim": act_dim,
        "activation": nn.PReLU,
        "input": {
            "type": "linear",
            "out": 64,
        },
        "output": {
            "in": 32,
        },
        "linear": {
            "num": 3,
            "ins": [64, 128, 64],
            "outs": [128, 64, 32],
        },
    }

    agent = DeepOptionsAgent(None, None, build_simple_network(fc_net_arch))

    print(agent.online)
    print(agent.target)
