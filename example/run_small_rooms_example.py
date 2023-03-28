import gc

import numpy as np
import matplotlib.pyplot as plt

from simpleoptions.option import Option
from simpleoptions import OptionAgent, PrimitiveOption

from small_rooms_env import SmallRoomsEnv
from small_rooms_doorway_option import DoorwayOption

if __name__ == "__main__":

    num_agents = 10

    results = []
    for run in range(num_agents):
        print(f"--- Run {run + 1} ---")

        # Initialise our environment.
        env = SmallRoomsEnv()

        # Add options corresponding to our environment's primitive actions.
        primitive_options = []
        for action in env.get_action_space():
            primitive_options.append(PrimitiveOption(action, env))
        env.options.extend(primitive_options)

        # Initialise our agent and train it for 50 episodes.
        agent = OptionAgent(env)
        n_episodes = 50
        result = agent.run_agent(n_episodes)
        results.append([sum(ep_rewards) for ep_rewards in result])

        gc.collect()

    primitive_results = np.array(results)

    results = []
    for run in range(num_agents):
        # Initialise our environment.
        env = SmallRoomsEnv()

        # Add options corresponding to our environment's primitive actions.
        primitive_options = []
        for action in env.get_action_space():
            primitive_options.append(PrimitiveOption(action, env))
        env.options.extend(primitive_options)

        # A an option for reaching the room's doorway.
        env.options.append(DoorwayOption())

        # Initialise our agent and train it for 50 episodes.
        agent = OptionAgent(env)
        n_episodes = 50
        result = agent.run_agent(n_episodes)
        results.append([sum(ep_rewards) for ep_rewards in result])

        gc.collect()

    option_results = np.array(results)

    # Plot a simple learning curve.
    x = list(range(n_episodes))
    plt.plot(x, np.mean(primitive_results, axis=0), label="Primtives", color="b")
    plt.fill_between(
        x,
        np.mean(primitive_results, axis=0) - np.std(primitive_results, axis=0),
        np.mean(primitive_results, axis=0) + np.std(primitive_results, axis=0),
        color="b",
        alpha=0.5,
    )
    plt.plot(x, np.mean(option_results, axis=0), label="Options", color="r")
    plt.fill_between(
        x,
        np.mean(option_results, axis=0) - np.std(option_results, axis=0),
        np.mean(option_results, axis=0) + np.std(option_results, axis=0),
        color="r",
        alpha=0.5,
    )
    plt.legend()
    plt.xlabel("Number of Episodes")
    plt.ylabel("Episode Return")
    plt.grid(which="both", axis="both", linestyle="-")
    plt.show()
