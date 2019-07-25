# BaRL-SimpleOptions
A small Python package which provides a simple framework to guide the use of options in reinforcement learning projects. Less of a plug-and-play repository, this code is intended to act as a basis to start your own work off of.
At the moment, the project is in its infancy, having been extracted and generalised from code used in a previous research project.
Extensions, including direct integration with OpenAI Gym-like environments, are planned.

# How to Install
Download the repository and, in the root folder, run the command `sudo python setup.py install`.

# How to use this code.
First, you should create an implementation of the `State` abstract class, according to the transition dynamics and requirements of whatever reinforcement learning environment you are using. How you represent actions, states etc. is up to you, as long as actions have a hashable representation (e.g. a string or integer).

You should then generate a [NetworkX](https://networkx.github.io/) `DiGraph` to represent the state transition dynamics of your environment. It should be simple enough to enumerate all possible states and transitions after implementing the `State` class. It may prove useful to override the `__str__` method of `State` so that every state has a unique string identifier, and then use that identifier as the id of the corresponding node on the graph.

It is then up to you how you discover options in your environment. Once you have done so, decide on an initiation set size for your options, and encode the option policy as a dictionary, with keys being state identifiers and values being the corresponding action. Save these dictionaries as .json files.

Finally, you can instantiate subgoal-directed options by supplying the NetworkX Digraph representing your environment, the path of a policy dictionary .json file, and your chosen initiation set size to the constructor of the `SubgoalOption` class. Primitive options can also easily be instantiated, by providing the identifier of a primitive action to the constructor of the `PrimitiveOption` class.
