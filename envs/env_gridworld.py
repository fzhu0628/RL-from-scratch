import numpy as np
class GridWorld:
    def __init__(self, grid_size=16, goal_state=(15, 15)):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.state = (0, 0)  # Start at the top-left corner

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.grid_size - 1)

        self.state = (x, y)
        reward = -5 if self.state != self.goal_state else 0
        done = self.state == self.goal_state
        return self.state, reward, done
    







