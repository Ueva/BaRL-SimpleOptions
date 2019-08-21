from barl_simpleoptions import BaseEnvironment
from two_rooms_state import TwoRoomsState

class TwoRoomsEnvironment(BaseEnvironment) :
    
    def __init__(self, options):
        super().__init__(options)
    
    def step(self, action) :
        # Work out next state.
        self.current_state = self.current_state.take_action(action)[0]

        # Work out reward - +10 if the goal is reached, -1 otherwise.
        if (self.current_state.is_terminal_state()) :
            reward = 10.0
            self.terminal = True
        else :
            reward = -1.0

        return self.current_state, reward, self.terminal

    def reset(self) :
        self.current_state = TwoRoomsState((0,0))
        self.terminal = False
        return self.current_state
