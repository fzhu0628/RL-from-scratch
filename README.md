# Implementation of Modern Reinforcement Learning (RL) Algorithms from Scratch

## Abstract
I am a PhD candidate in the *Department of Electrical and Computer Engineering* at North Carolina State University. My research focuses on the theory of reinforcement learning (RL), particularly on leveraging multi-agent structures to improve sample efficiency. Specifically, we develop algorithms with provable guarantees to demonstrate the benefits of collaboration and decentralized learning.

While much of our work is theoretical, this project aims to bridge theory and practice by implementing modern RL algorithms from scratch, including Q-learning, DQN, DDPG, TRPO, and PPO. In addition, we explore extensions inspired by multi-agent RL to evaluate their empirical performance across different environments. This is an evolving project intended to provide both practical insights and theoretical intuition.

---

## Value-Based Algorithms

In this section, we implement three representative value-based methods:
- Tabular Q-learning  
- Q-learning with Linear Function Approximation (LFA)  
- Deep Q-Learning (DQN)  

These methods illustrate the progression from exact representations to function approximation and deep learning.

---

### Tabular Q-Learning

Tabular Q-learning, originally proposed by Christopher Watkins, is one of the most fundamental algorithms in reinforcement learning. It maintains an explicit table of Q-values for each state-action pair and updates them using stochastic approximations of the Bellman optimality operator.

Each update is given by:

$$
Q_{t+1}(s,a) = (1-\alpha) Q_t(s,a) + \alpha \left(r + \gamma \max_{a'} Q_t(s',a')\right)
$$

Under standard assumptions, tabular Q-learning converges to the optimal action-value function $Q^*$ with a rate of $\tilde{O}(1/\sqrt{T})$. Its simplicity and strong theoretical guarantees make it a natural starting point for understanding more advanced methods.

**Implementation:** `algos/q_learning.py`

---

### Q-Learning with Linear Function Approximation (LFA)

To handle larger or continuous state spaces, tabular representations become infeasible. A natural extension is to approximate the Q-function using a linear model:

$$
Q(s,a; \theta) = \theta^\top \phi(s,a)
$$

where $\phi(s,a)$ is a feature representation of the state-action pair.

This approach enables generalization across states while significantly reducing memory requirements. However, unlike the tabular setting, the parameters $\theta$ are shared across all state-action pairs. As a result, updates are no longer local, and the algorithm effectively performs a projected Bellman update:

$$
Q \approx \Pi T Q
$$

where $\Pi$ denotes projection onto the function class.

This introduces several challenges:
- Loss of contraction properties  
- Potential instability or divergence  
- Sensitivity to feature design  

In our implementation, we explore simple feature mappings such as normalized state coordinates and one-hot encodings of actions.

**Implementation:** `algos/q_learning_lfa.py`

---

### Deep Q-Learning (DQN)

Deep Q-Networks (DQN) replace hand-crafted features with neural networks:

$$
Q(s,a;\theta) = \text{NN}_\theta(s)[a]
$$

where the network outputs Q-values for all actions given a state.

While this significantly improves representational power, it introduces instability due to the combination of:
- bootstrapping  
- function approximation  
- off-policy learning  

To address these issues, DQN incorporates several key techniques:

#### Experience Replay
A replay buffer stores past transitions and samples mini-batches uniformly, reducing temporal correlations and approximating i.i.d. sampling.

#### Target Network
A separate network with frozen parameters is used to compute the target:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

This stabilizes training by decoupling the target from the rapidly changing Q-network.

#### Mini-batch Updates
Gradient-based optimization is performed over batches of transitions for improved stability and efficiency.

DQN can be viewed as a nonlinear extension of linear function approximation, where the feature representation is learned jointly with the value function.

**Implementation:** `algos/dqn.py`