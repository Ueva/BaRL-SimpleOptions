from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete

from simpleoptions.function_approximation import ApproxBaseEnvironment


class GymEnv(ApproxBaseEnvironment):
    def __init__(self, gym_env: Env):
        self.gym_env = gym_env
        self.done = True

    def reset(self, state=None):
        if state is not None:
            raise ValueError("GymEnv does not support initial state specification.")

        initial_state, info = self.gym_env.reset()
        self.current_state = initial_state
        self.done = False
        return initial_state, info

    def step(self, action, state=None):
        assert state is None

        next_state, reward, terminated, truncated, info = self.gym_env.step(action)
        self.current_state = next_state
        self.done = terminated or truncated
        return next_state, reward, terminated, truncated, info

    def render(self, render_mode="human"):
        return self.gym_env.render()

    def close(self):
        return self.gym_env.close()

    def get_state_space(self):
        return self.gym_env.observation_space

    def get_action_space(self):
        return self.gym_env.action_space

    def is_state_terminal(self, state=None):
        """
        Returns whether or not the **current state** is terminal.
        Note: GymEnv.is_state_terminal does not support the use of the `state` parameter.

        Args:
            state (Hashable, optional): Unused. Defaults to None.

        Raises:
            ValueError: Raises a ValueError if a state is specified.

        Returns:
            bool: Whether the **current state** is terminal.
        """
        if state is not None:
            raise ValueError("GymEnv does not support state specification.")

        return self.done

    def get_available_actions(self, state=None):
        """
        Returns the actions available in the **current state**.
        Note: GymEnv.get_available_actions does not support the use of the `state` parameter.

        Args:
            state (Hashable, optional): Unused. Defaults to None.

        Raises:
            ValueError: Raises a ValueError if a state is specified.

        Returns:
            List: Whether the **current state** is terminal.
        """
        if state is not None:
            raise ValueError("GymEnv does not support state specification.")

        if type(self.get_action_space()) == Discrete:
            return self.get_action_space().n
        elif type(self.get_action_space()) == MultiDiscrete:
            return self.get_action_space().nvec
