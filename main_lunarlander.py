import gymnasium as gym
from algos.REINFORCE import reinforce, run_reinforce_lunarlander
from algos.DQN import dqn_interleave_buffer_and_training
from algos.A2C import a2c, run_a2c_lunarlander, a2c_multi_env
from envs.env_lunarlander import LunarLander

def make_envs(env_id, n_envs):
    return gym.vector.SyncVectorEnv([
        lambda: gym.make(env_id) for _ in range(n_envs)
    ])
envs = make_envs("LunarLander-v3", n_envs=16)
# env = LunarLander(render=False)
# model = reinforce(env)
# model = dqn_interleave_buffer_and_training(env)
model = a2c_multi_env(envs, rounds=40000, n_steps=20, gamma=0.99, lr=1e-4)

env = LunarLander(render=False)
for _ in range(5):
    run_a2c_lunarlander(env, model)
