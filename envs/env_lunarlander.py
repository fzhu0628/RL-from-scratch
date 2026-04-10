import gymnasium as gym
import numpy as np

class LunarLander:
    def __init__(self, render=False):
        if render:
            self.env = gym.make("LunarLander-v3", render_mode="human")
        else:
            self.env = gym.make("LunarLander-v3")

        # properties (for consistency with your GridWorld API)
        self.state_dim = 8
        self.action_dim = 4

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        return next_state, reward, terminated, truncated

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()