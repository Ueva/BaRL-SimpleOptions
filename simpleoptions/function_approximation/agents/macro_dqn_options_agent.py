import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simpleoptions import BaseOption
from simpleoptions.function_approximation import ApproxBaseEnvironment, PrimitiveOption

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, DefaultDict
from numpy.random import Generator as RNG

import random

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simpleoptions.function_approximation.agents.replay_buffer import ReplayBuffer, PrimitiveTransition

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, DefaultDict, List


class MacroDQN:
    def __init__(
        self,
        env: gym.Env,
        test_env: gym.Env,
        network: nn.Module,
        alpha: float,
        epsilon: float,
        gamma: float,
        tau: float,
        buffer_capacity: int,
        batch_size: int,
        num_updates: int,
        rng: RNG = None,
    ):
        super().__init__()

        # Initialise random number generator.
        self.rng = rng if rng else random

        # Set environment variables.
        self.env = env
        self.test_env = test_env

        # Build online q-network for action selection.
        self.online = deepcopy(network)

        # Build the target q-network for computing td targets.
        self.target = deepcopy(network)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        # Initialise optimisers.
        self.optimiser = torch.optim.RAdam(self.online.parameters(), lr=alpha)

        # Initialise replay memory.
        self.buffer = ReplayBuffer(buffer_capacity)

        # Initialise option execution tracking lists.
        self.executing_options = []
        self.executing_options_states = []
        self.executing_options_rewards = []

        # Initialise hyperparameters.
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_updates = num_updates

    def select_action(
        self, state, executing_options: List["BaseOption"], test=False
    ):  # TODO: Implement option masking based on initiation sets.
        # If no options are currently executing, select an action using the online network.
        if len(executing_options) == 0:
            # Get currently available options.
            available_options_idx = self.env.get_available_options(state, get_indices=True)

            if self.rng.random() < self.epsilon and test == False:
                return self.rng.choice(available_options_idx)
            else:
                with torch.no_grad():
                    vals = self.online.forward(state)
                    mask = torch.tensor(
                        [[1 if i in available_options_idx else -float("inf") for i in range(self.env.option_space.n)]]
                    )
                    action = torch.argmax(vals * mask).item()
                    return action
        # Otherwise, select according to the currently executing option's policy.
        else:
            return executing_options[-1].policy(state, test)

    def update_online_network(self):
        for _ in range(self.num_updates):
            # Sample a batch of transitions from the replay buffer.
            sampled_transitions = self.buffer.sample(self.batch_size)
            batch = PrimitiveTransition(*zip(*sampled_transitions))

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Mask for terminal/non-terminal next states.
            non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])

            # Get current action-value estimates.
            preds = self.online(state_batch).gather(1, action_batch)

            # Compute TD targets.
            next_state_values = torch.zeros(self.batch_size, dtype=torch.float32)
            with torch.no_grad():
                next_state_values[non_terminal_mask] = self.target(non_terminal_next_states).max(1).values
            targets = reward_batch + self.gamma * next_state_values

            # Compute loss.
            loss_function = nn.HuberLoss()
            loss = loss_function(preds, targets.unsqueeze(1))

            # Update online network.
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

    @torch.no_grad()
    def update_target_network(self):
        tau = self.tau
        for p_o, p_t in zip(self.online.parameters(), self.target.parameters()):
            p_t.data.copy_(tau * p_o.data + (1 - tau) * p_t.data)
            p_t.requires_grad = False

    def run_agent(
        self,
        num_epochs: int,
        epoch_length: int,
        render_interval: int = 0,
        test_interval: int = 0,
        test_length: int = 0,
        test_runs: int = 0,
        verbose_logging: bool = False,
        episodic_eval: bool = False,
    ) -> Tuple[DefaultDict, DefaultDict | None]:

        # Set the time-step limit.
        num_time_steps = num_epochs * epoch_length

        # If we are resting the greedy policy separately, make a separate copy of
        # the environment to use for those tests. Also initialise variables for
        # tracking test performance.
        training_rewards_ = [None for _ in range(num_time_steps)]

        if test_interval > 0:
            test_interval_time_steps = test_interval * epoch_length
            evaluation_rewards = [None for _ in range(num_time_steps // test_interval_time_steps)]

        episodes = 0
        time_steps = 0

        while time_steps < num_time_steps:
            # Initialise state variables.
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminal = False
            truncated = False

            # Render initial state.
            if render_interval > 0:
                self.env.render()

            while not terminal or truncated:
                # Select an action.
                option_index = self.select_action(state, self.executing_options)
                option = self.env.index_to_option[option_index]

                # Handle if the selected option is a primitive option.
                if isinstance(option, PrimitiveOption):
                    self.executing_options.append(option)
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])

                    time_steps += 1

                    observation, reward, terminal, _, _ = self.env.step(option.action)
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                    # Render, if we need to.
                    if render_interval > 0 and time_steps % render_interval == 0:
                        self.env.render()

                    state = next_state

                    # Add next state and reward to the executing option tracking lists.
                    for i in range(len(self.executing_options)):
                        self.executing_options_states[i].append(state)
                        self.executing_options_rewards[i].append(reward)

                    # Terminate any options that need terminating this time-step.
                    while self.executing_options and self._roll_termination(self.executing_options[-1], next_state):
                        if self.executing_options[-1] not in self.env.exploration_options:
                            self.add_experience_to_buffer(
                                self.executing_options[-1],
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                terminal,
                            )
                        self.executing_options.pop()
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()

                    # Perform updates.
                    if len(self.buffer) > self.batch_size:
                        self.update_online_network()
                        self.update_target_network()

                # Handle if the selected option is a higher-level option.
                elif isinstance(option, BaseOption):
                    self.executing_options.append(option)
                    self.executing_options_states.append([state])
                    self.executing_options_rewards.append([])
                else:
                    RuntimeError("Invalid option type.")

                # Terminate if we have exceeded the maximum number of time steps.
                if time_steps >= num_time_steps:
                    truncated = True

                # Handle if the current state is terminal.
                if terminal or truncated:
                    while len(self.executing_options) > 0:
                        if self.executing_options[-1] not in self.env.exploration_options:
                            self.add_experience_to_buffer(
                                self.executing_options[-1],
                                self.executing_options_states[-1],
                                self.executing_options_rewards[-1],
                                terminal,
                            )
                        self.executing_options.pop()
                        self.executing_options_states.pop()
                        self.executing_options_rewards.pop()

            episodes += 1

    def add_experience_to_buffer(self, option, states, rewards, terminal):
        option_index = self.env.option_to_index[option]

        discounted_returns = [sum([self.gamma**i * r for i, r in enumerate(rewards[j:])]) for j in range(len(rewards))]

        if not terminal:
            terminating_state = states[-1]
        else:
            terminating_state = None

        trajectory_length = len(states)
        for i in range(trajectory_length - 1):
            self.buffer.add(
                states[i],
                torch.tensor([[option_index]], dtype=torch.long),
                torch.tensor([discounted_returns[i]], dtype=torch.float32),
                terminating_state,
                False if i < trajectory_length - 1 else terminal,
            )

    def _roll_termination(self, option: "BaseOption", state):
        # Rolls on whether or not the given option terminates in the given state.
        # Will work with stochastic and deterministic termination functions.
        if self.rng.random() > option.termination(state):
            return False
        else:
            return True


if __name__ == "__main__":
    from torchinfo import summary

    from simpleoptions.function_approximation.utils.network_builders import CriticNetworkFC
    from simpleenvs.envs.continuous_rooms import ContinuousFourRooms
    from simpleoptions.function_approximation import GymWrapper

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    torch.set_default_device(device)

    env = GymWrapper(ContinuousFourRooms(render_mode="human"))
    test_env = GymWrapper(ContinuousFourRooms(render_mode="human"))

    primitive_options = [PrimitiveOption(i) for i in range(env.action_space.n)]

    env.set_options(primitive_options)
    test_env.set_options(primitive_options)

    obs_dim = len(env.observation_space.sample().flatten())
    act_dim = env.option_space.n

    agent = MacroDQN(
        env,
        test_env,
        network=CriticNetworkFC(obs_dim, act_dim),
        alpha=0.001,
        epsilon=0.15,
        gamma=0.9999,
        tau=0.01,
        buffer_capacity=100_000,
        batch_size=8,
        num_updates=1,
    )

    print(agent.online)
    print(agent.target)
    summary(agent.online, input_size=(1, obs_dim), device=device)
    print(f"obs_dim: {obs_dim}, act_dim: {act_dim}")

    agent.run_agent(num_epochs=100, epoch_length=100, render_interval=10)
