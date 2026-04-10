import numpy as np
from envs.env_cartpole import CartPoleEnv
from algos.DQN import dqn_interleave_buffer_and_training, dqn_separate_buffer_and_training, run_dqn_policy_cartpole
from algos.REINFORCE import reinforce, run_reinforce_lunarlander
import gymnasium as gym
from algos.RAINBOW import rainbow
from algos.A2C import a2c, run_a2c_lunarlander

env = CartPoleEnv(render=False)
# model = dqn_separate_buffer_and_training(env)
# model = dqn_interleave_buffer_and_training(env)
# model = rainbow(env)
# model = reinforce(env)
model = a2c(env, lr=1e-3)

# env = CartPoleEnv(render=False)
# reward = run_dqn_policy_cartpole(env, model)
reward = run_a2c_lunarlander(env, model)

print("Episode reward:", reward)

# env = gym.make("CartPole-v1", render_mode="rgb_array")
# from gymnasium.wrappers import RecordVideo

# env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)