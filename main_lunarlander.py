import gymnasium as gym
from algos.REINFORCE import reinforce, run_reinforce_lunarlander
from algos.DQN import dqn_interleave_buffer_and_training
from algos.A2C import a2c, run_a2c_lunarlander, a2c_multi_env
from algos.PPO import ppo_multi_env
from envs.env_lunarlander import LunarLander
import matplotlib.pyplot as plt

def make_envs(env_id, n_envs):
    return gym.vector.SyncVectorEnv([
        lambda: gym.make(env_id) for _ in range(n_envs)
    ])
envs = make_envs("LunarLander-v3", n_envs=16)
# env = LunarLander(render=False)
# model = reinforce(env)
# model = dqn_interleave_buffer_and_training(env)
model_a2c, reward_history_a2c = a2c_multi_env(envs, rounds=40000, n_steps=20, gamma=0.99, lr=1e-4)
model_ppo, reward_history_ppo = ppo_multi_env(envs, rounds=40000, n_steps=20, gamma=0.99, lr=1e-4, epsilon=0.2, m_train=4)

env = LunarLander(render=False)

# %%
plt.figure()
plt.plot(reward_history_a2c, label="A2C")
plt.plot(reward_history_ppo, label="PPO")
plt.xlabel("Update Step")
plt.ylabel("Policy Change (illustrative)")
plt.title("A2C vs PPO Training Dynamics")
plt.legend()
plt.grid()
plt.show()


for _ in range(5):
    run_a2c_lunarlander(env, model_a2c)
    print("----------------------------------")
    run_a2c_lunarlander(env, model_ppo)
