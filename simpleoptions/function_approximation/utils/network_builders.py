import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict


def build_simple_network(net_arch: Dict):
    network = nn.Sequential()

    # Input layer.
    if net_arch["input"]["type"] == "linear":
        network.append(
            nn.Linear(
                in_features=net_arch["observation_dim"],
                out_features=net_arch["input"]["out"],
            )
        )
    elif net_arch["input"]["type"] == "conv2d":
        network.append(
            nn.Conv2d(
                in_channels=net_arch["observation_dim"],
                out_channels=net_arch["input"]["out"],
                kernel_size=net_arch["input"]["kernel_size"],
                stride=net_arch["input"]["stride"],
            )
        )
    else:
        raise ValueError(f"Input layer type should be either `linear` or `conv2d`, not `{net_arch['input']['type']}`.")
    network.append(net_arch["activation"]())

    # Add convolutional layers.
    if "cnn" in net_arch and "num" in net_arch["cnn"] and net_arch["cnn"]["num"] > 0:
        for i in range(net_arch["cnn"]["num"]):
            network.append(
                nn.Conv2d(
                    in_channels=net_arch["cnn"]["ins"][i],
                    out_channels=net_arch["cnn"]["outs"][i],
                    kernel_size=net_arch["cnn"]["kernel_sizes"][i],
                    stride=net_arch["cnn"]["strides"][i],
                )
            )
            network.append(net_arch["activation"]())
        network.append(nn.Flatten())

    # Add fully-connected (linear) layers.
    if "linear" in net_arch and "num" in net_arch["linear"] and net_arch["linear"]["num"] > 0:
        for i in range(net_arch["linear"]["num"]):
            network.append(
                nn.Linear(
                    net_arch["linear"]["ins"][i],
                    net_arch["linear"]["outs"][i],
                )
            )
            network.append(net_arch["activation"]())

    # Output layer.
    network.append(
        nn.Linear(net_arch["output"]["in"], net_arch["action_dim"]),
    )

    return network


## Example Inputs for `build_simple_network`:
# cnn_net_arch = {
#     "observation_dim": obs_dim,
#     "action_dim": act_dim,
#     "activation": nn.PReLU,
#     "input": {
#         "type": "conv2d",
#         "out": 32,
#         "kernel_size": 8,
#         "stride": 4,
#     },
#     "output": {
#         "in": 128,
#     },
#     "cnn": {
#         "num": 2,
#         "ins": [32, 64],
#         "outs": [64, 64],
#         "kernel_sizes": [4, 3],
#         "strides": [4, 1],
#     },
#     "linear": {
#         "num": 2,
#         "ins": [2048, 512],
#         "outs": [512, 128],
#     },
# }

# fc_net_arch = {
#     "observation_dim": obs_dim,
#     "action_dim": act_dim,
#     "activation": nn.PReLU,
#     "input": {
#         "type": "linear",
#         "out": 64,
#     },
#     "output": {
#         "in": 32,
#     },
#     "linear": {
#         "num": 3,
#         "ins": [64, 128, 64],
#         "outs": [128, 64, 32],
#     },
# }


class CriticNetworkFC(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.f1 = torch.nn.Linear(obs_dim, 64)
        self.prelu1 = torch.nn.PReLU()
        self.f2 = torch.nn.Linear(64, 128)
        self.prelu2 = torch.nn.PReLU()
        self.f3 = torch.nn.Linear(128, 64)
        self.prelu3 = torch.nn.PReLU()
        self.critic_out = torch.nn.Linear(64, act_dim)

    def forward(self, x):

        x = self.f1(x)
        x = self.prelu1(x)
        x = self.f2(x)
        x = self.prelu2(x)
        x = self.f3(x)
        x = self.prelu3(x)
        x = self.critic_out(x)

        return x


class CriticNetworkCNN(nn.Module):
    pass


class DuellingCriticNetworkFC(nn.Module):
    pass


class DuellingCriticNetworkCNN(nn.Module):
    pass


class PolicyNetworkFC(nn.Module):
    pass


class PolicyNetworkCNN(nn.Module):
    pass
