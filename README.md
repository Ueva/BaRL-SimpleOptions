# SimpleOptions

This Python package aims to provide a simple framework for implementing and using options in Hierarchical Reinforcement Learning (HRL) projects.

Key classes:

- `BaseOption`: An abstract class representing an option with an initiation set, option policy, and termination condition.
- `BaseEnvironment`: An abstract base class representing an agent's environment. The environment specification is based on the OpenAI Gym `Env` specifciation, but does not implement it directly. It supports both primitive actions and options, as well as functionality for constructing State-Transition Graphs (STGs) out-of-the-box using NetworkX.
- `OptionAgent`: A class representing an HRL agent, which can interact with its environment and has access to a number of options. It includes implementations of Macro-Q Learning and Intra-Option learning, with many customisable features.

This code was written with tabular, graph-based HRL methods in mind. It's less of a plug-and-play repository, and is intended to be used to as a basic framework for developing your own `BaseOption` and `Environment` implementations.

## How to Install

The easiest way to install this package is to simply run `pip install simpleoptions`.

Alternatively, you can install from source. Simply download this repository and, in the root directory, run the command `pip install .`

## How to Use This Code

Below, you will find a step-by-step guide introducing the intended workflow for using this code.

### Step 1: Implement an Environment

The first step to using this framework involves defining an environment for your agents to interact with. This can be done by subclassing the `BaseEnvironment` abstract class and filling in the abstract methods. If you have previously worked with OpenAI Gym/Farama Gymnasium, much of this will be familiar to you, although there are a few additional methods on top of the usual `step` and `reset` that you'll need to implement.

### Step 2: Define Your Options

You must now define/discover options for your agent to use when interacting with its environment. How you go about this is up to you. An ever-growing number of option discovery methods can be found in the hierarchical reinforcement learning literature. We include some option discovery method implementations in the `implementations` directory.

To define a new type of option, you need to subclass `BaseOption` and implement the following methods:

- `initiation` - a method that takes a state as its input, and returns whether the option can be invoked in that state.
- `termination` - a method that takes a state as its input, and returns the probability that the option terminates in that state.
- `policy` - a method that takes a state as its input, and returns the action (either a primitive action or another option) that this option would select in this state.

This minimal framework gives you a lot of flexibility in defining your options. For example, your `policy` method could make use of a simple dictionary mapping states to actions, it could be based on some learned action-value function, or any other function of the state.

As an example, consider an example option that takes an agent to a sub-goal state from any of the nearest 50 states. `initiation` would return `True` for the nearest 50 states to the subgoal, and `False` otherwise. `termination` would return `0.0` for states in the initiation set, and `1.0` otherwise. `policy` woudl return the primitive action that takes the agent one step along the shortest path to the subgoal state.

Finally, we also include a `PrimitiveOption` that can be used to represent the primitive actions made available by a given environment.

### Step 5: Giving Options to an Agent and Running It in an Environment

This package also includes an `OptionsAgent`, an implementation of an agent that learns using the Macro-Q Learning and Intra-Option Learning algorithms.

Once you have defined an environment an a set of options, you can instatiate an `OptionsAgent` agent and use its `run_agent` method to train it.

## Example Environments

A number of reinforcement learning environments implemented using our `BaseEnvironment` interfaces can be found [here](https://github.com/Ueva/BaRL_Envs).
