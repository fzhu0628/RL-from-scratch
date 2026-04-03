import gymnasium as gym
import numpy as np

class CartPoleEnv:
    def __init__(self, render=False):
        if render:
            self.env = gym.make("CartPole-v1", render_mode="human")
        else:
            self.env = gym.make("CartPole-v1")

        # properties (for consistency with your GridWorld API)
        self.state_dim = 4
        self.action_dim = 2

    def reset(self):
        state, _ = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

