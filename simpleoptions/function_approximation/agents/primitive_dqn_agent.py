import random

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from simpleoptions.function_approximation.agents.replay_buffer import ReplayBuffer, PrimitiveTransition

from copy import deepcopy
from collections import defaultdict
from typing import Dict, Tuple, DefaultDict


class DQN:
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
    ):
        super().__init__()

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

        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.num_updates = num_updates

    def select_action(self, state, test=False):
        if random.random() < self.epsilon and test == False:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.online.forward(state)).item()

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

    def warm_up(self, warmup_length: int):

        warmup_time_step = 0
        while warmup_time_step < warmup_length:

            # Initialise state variables.
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            terminal = False

            while not terminal:
                # Randomly select an action.
                action = torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

                # Execute the action, observe the next state and reward.
                observation, reward, terminal, _, _ = self.env.step(action.item())

                if not terminal:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                else:
                    next_state = None

                reward = torch.tensor([reward], dtype=torch.float32)

                # Add experience to the replay buffer.
                self.buffer.add(state, action, reward, next_state, terminal)

                state = next_state

                warmup_time_step += 1

                if warmup_time_step >= warmup_length:
                    terminal = True

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
        warmup_length: int = 0,
    ) -> Tuple[DefaultDict, DefaultDict | None]:

        if warmup_length > 0:
            self.warm_up(warmup_length)

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

            # Render initial state.
            if render_interval > 0:
                self.env.render()

            while not terminal:
                # Select an action.
                action = torch.tensor([[self.select_action(state)]], dtype=torch.long)

                # Execute the action, observe the next state and reward.
                observation, reward, terminal, _, _ = self.env.step(action.item())

                if not terminal:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                else:
                    next_state = None

                reward = torch.tensor([reward], dtype=torch.float32)

                # Add experience to the replay buffer.
                self.buffer.add(state, action, reward, next_state, terminal)

                state = next_state

                # Perform updates.
                self.update_online_network()
                self.update_target_network()

                time_steps += 1

                # Render initial state.
                if render_interval > 0 and time_steps % render_interval == 0:
                    self.env.render()

                if time_steps >= num_time_steps:
                    terminal = True

            episodes += 1


if __name__ == "__main__":
    from torchinfo import summary

    from simpleoptions.function_approximation.utils.network_builders import CriticNetworkFC
    from simpleenvs.envs.continuous_rooms import ContinuousFourRooms

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    torch.set_default_device(device)

    # env = gym.make("Acrobot-v1")
    # test_env = gym.make("Acrobot-v1")
    env = ContinuousFourRooms(render_mode="human")
    test_env = ContinuousFourRooms(render_mode="human")
    obs_dim = len(env.observation_space.sample().flatten())
    act_dim = env.action_space.n

    agent = DQN(
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

    agent.run_agent(num_epochs=100, epoch_length=100, warmup_length=1024, render_interval=0)
