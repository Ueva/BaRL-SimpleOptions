import networkx as nx
import matplotlib.pyplot as plt

from barl_simpleoptions import SubgoalOption
from barl_simpleoptions import PrimitiveOption
from barl_simpleoptions import OptionAgent

from two_rooms_environment import TwoRoomsEnvironment
from two_rooms_state import TwoRoomsState



################################
## Generate Interaction Graph ##
################################

# Generate state-interaction graph for this environment and save it to a file.
initial_state = TwoRoomsState((0,0))
state_transition_graph = initial_state.generate_interaction_graph([initial_state])
nx.write_gexf(state_transition_graph, "sa_graph.gexf")



########################
## Construct options. ##
########################
options = []

# Construct primitive options.
primitive_actions = TwoRoomsState.actions
for action in primitive_actions :
    options.append(PrimitiveOption(action))

# Construct subgoal-directed option (i.e. door subgoal).
door_policy_file_path = "door_option_policy.json"
door_option = SubgoalOption(TwoRoomsState((1,3)), state_transition_graph, door_policy_file_path, 19)
options.append(door_option)



####################################
## Initialise Environment & Agent ##
####################################
env = TwoRoomsEnvironment(options)

# Instantiate agent.
agent = OptionAgent(env)



#################
### Run Agent ###
#################

# Run agent for 50 episodes.
num_episodes = 50
episode_returns = agent.run_agent(num_episodes)

# Save Return Graph.
plt.plot(range(1, num_episodes + 1), episode_returns)
plt.title("Agent Training Curve")
plt.xlabel("Episode Number")
plt.ylabel("Episode Return")
plt.savefig("episode_returns.png")