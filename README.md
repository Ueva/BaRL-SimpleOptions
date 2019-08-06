# BaRL-SimpleOptions

A small Python package which provides a simple framework to guide the use of options in reinforcement learning projects. Less of a plug-and-play repository, this code is intended to act as a basis to start your own work off of.
At the moment, the project is in its infancy, having been extracted and generalised from code used in a previous research project.
Extensions, including direct integration with OpenAI Gym-like environments, are planned.

## How to Install

Download the repository and, in the root folder, run the command `sudo python setup.py install`.

## How to Use This Code

There is an intended workflow to using this framework, if you do not just want to try to slot it into your own existing code. Below, this intended workflow is described in a step-by-step manner.

### Step 1: Implement Your Environment

The first step to using this framework involves defining the dynamics of your chosen reinforcement learning environment. This can be done by implementing the `State` abstract class. A number of method interfaces have been provided, and implementing them will allow you to use your environment with the rest of this framework. Some are optional, and so have not been marked as abstract methods, though you should think about whether implementing them might be useful for you.

### Step 2: Graph the State-Space

With your environment's state-transition dynamics defined in your implementation of the `State` class, you are now ready to create a graph of your reinforcement learning environment. This graph will encode states as nodes, and possible state-transitions as edges.

BaRL-SimpleOptions makes use of the [NetworkX](https://networkx.github.io/) package for representing graphs. Creating the NetworkX `DiGraph` to represent the interaction graph of your environment can be easily achieved by recursively enumerating successors of states, starting with a small set of initial states. You may need to make use of methods such as `get_successors`, `get_available_actions`, take_action and `is_terminal_state` from your implementation of the `State` class.

### Step 3: Generate Options

You must now discover options for your environment. How you go about this is up to you (this framework helps you use options, not discover them). An ever-growing number of option discovery techniques can be found in the reinforcement learning literature.

An option is made up of the following:

- An initiation set - states in which the option can be invoked.
- A termination condition - returns true when the option should terminate.
- An option policy - a policy to be executed while the option is active.

For each option, you should record encode its policy as dictionary mapping states (i.e. the string representation of a state) to actions, and save this dictionary as a .json file.

### Step 4: Instantiate Options

Finally, you can instantiate your options. We include an abstract base class, `Option`, from which you can define your own option types. All option subclasses should define at least an `initiation`, `termination` and `policy` method, to perform the function described above.

We also include two pre-defined subclasses of `Option` - `PrimitiveOption` and `SubgoalOption`.

- `PrimitiveOption` encodes a primitive action as an option - its initiation set contains all states where its underlying primitive action is available, it terminates with probability one in all states, and has the executes its primitive action as its policy.
- `SubgoalOption` encodes an option which represents working towards some subgoal state - its initiation set consists of the `n` closest states to the subgoal state, it terminates when leaving the initiation set or reaching the subgoal, and its policy should be an optimal policy for reaching the subgoal state (this policy should be found and supplied as a dictionary  saved to a .json file, as described above).

These two types of option are used widely in the literature, and should act as a starting-point for implementing your own custom option types.

### Step 5: Running an Agent Which Uses Options

If you do not want to make use of our options class with your own agent and environment code, our framework also includes a simple implementation of an agent which learns to act using options in an environment, and a base class from which to implement an environment for the agent to act in. This agent learns using the macro-Q and intra-option learning algorithms.

Our `Environment` and `BaseEnvironment` base classes are inspired by (and *may*, in the future, be integrated with) OpenAI Gym environments. Using functionality from you previously defined `State` implementation, you should be able to quite easily be able to implement an environment for an agent to interact with.

Once you have defined your environment, you can run an agent using it by instantiating the `OptionAgent` class with your environment, and calling the `run_agent` method.
