# Implementation of Modern Reinforcement Learning (RL) Algorithms from Scratch
**Abstract** I am a PhD candidate in the *Department of Electrical and Computer Engineering* in *North Carolina State University*. My research focus is *Reinforcement Learning (RL) theory*, where we leverage the multi-agent structure to improve sample efficiency of the system. Speicifically, we develop algorithms along with provable guarantees to demonstrate the benefits of collaboration. While a majority of our work is theoretical, in this project, we plan to implement modern RL algorithms from scratch such as Q-learning, DQN, DDPG, TRPO, PPO, etc. We might also add a few experiments incorporating the ideas of our multi-agent algorithms to test their performance in multiple environments. This is a long-time-evolving project and I hope it brings insights to both us and the readers.

## Value-Based Algorithms
In this section, we implement three algorithms: tabular Q-learning, Q-learning with linear function approximation (LFA) and deep Q-learning (DQN).

### Tabular Q-Learning
This is the most fundamental algorithm in RL, proposed by Christopher Watkins. Using a convex combination of the old Q-table and the Q-table operated by an estimate of the Bellman optimality operator, the tabular Q-learning algorithm is proved to converge to the optimum at a rate of $\tilde{O} (1/\sqrt{T})$. This algorithm is implemented in 'algos/q_learning.py'.

### Q-Learning with LFA
