import numpy as np
from envs.env_cartpole import CartPoleEnv
from algos.DQN import dqn_interleave_buffer_and_training, dqn_separate_buffer_and_training, run_dqn_policy_cartpole
import gymnasium as gym
from algos.RAINBOW import rainbow

env = CartPoleEnv(render=False)
model = dqn_separate_buffer_and_training(env)
# model = dqn_interleave_buffer_and_training(env)
# model = rainbow(env)

# env = CartPoleEnv(render=False)
reward = run_dqn_policy_cartpole(env, model)

print("Episode reward:", reward)

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# from gymnasium.wrappers import RecordVideo

# env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)