# Implementation of Modern Reinforcement Learning (RL) Algorithms from Scratch

## Abstract
I am a PhD candidate in the *Department of Electrical and Computer Engineering* at North Carolina State University. My research focuses on the theory of reinforcement learning (RL), particularly on leveraging multi-agent structures to improve sample efficiency. Specifically, we develop algorithms with provable guarantees to demonstrate the benefits of collaboration and decentralized learning.

While much of our work is theoretical, this project aims to bridge theory and practice by implementing modern RL algorithms from scratch, including Q-learning, DQN, DDPG, TRPO, PPO, etc. In addition, we explore extensions inspired by multi-agent RL to evaluate their empirical performance across different environments. This is an evolving project intended to provide both practical insights and theoretical intuition.

---

## Environments

We evaluate our implementations on a set of environments with increasing complexity, starting from simple tabular settings and progressing to continuous-state control problems.

### GridWorld

GridWorld is a discrete environment where the agent navigates a 2D grid to reach a goal state. The state space consists of grid coordinates, and the action space includes four discrete actions (up, down, left, right).

This environment is useful for:
- validating correctness of tabular algorithms
- understanding convergence behavior
- debugging implementations

However, due to its small and fully observable state space, it does not require function approximation and therefore does not fully demonstrate the advantages of deep RL methods.

---

### CartPole

CartPole is a classic control problem where the agent must balance a pole on a moving cart. The state space is continuous and consists of position, velocity, angle, and angular velocity, while the action space is discrete (left or right force).

This environment introduces:
- continuous state representation
- need for function approximation
- more realistic control dynamics

It serves as a natural next step for evaluating linear function approximation and deep Q-learning methods.

---

## Value-Based Algorithms

In this section, we implement three representative value-based methods:
- Tabular Q-learning  
- Q-learning with Linear Function Approximation (LFA)  
- Deep Q-Networks (DQN)  

These methods illustrate the progression from exact representations to function approximation and deep learning.

---

### Tabular Q-Learning

Tabular Q-learning, originally proposed by Christopher Watkins, is one of the most fundamental algorithms in RL. It maintains an explicit table of Q-values for each state-action pair and updates them using stochastic approximations of the Bellman optimality operator.

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

In our implementation, we explore simple feature mappings such as normalized state coordinates and one-hot encodings of actions. In practice, however, designing effective features is far from trivial. Even in a small GridWorld of size $4 \times 4$, a low-dimensional feature vector (e.g., dimension 15) is often insufficient to accurately represent the action-value function, especially when compared to the total number of state-action pairs ($4 \times 4 \times 4 = 64$).

More importantly, the limitation is not merely the dimensionality, but the expressive power of the chosen features. Poorly designed features often fail to capture key structural properties of the environment (e.g., spatial relationships or boundary effects), leading to suboptimal performance and unstable learning dynamics. From a theoretical perspective, even in the linear function approximation setting, Q-learning is not guaranteed to converge and may diverge due to the loss of the contraction property of the Bellman operator under projection.

As a result, Q-learning with LFA is generally limited to relatively simple tasks and requires careful feature engineering. Nevertheless, it provides an important conceptual bridge between tabular methods and deep RL, where feature representations are learned **automatically** via neural networks.

**Implementation:** `algos/q_learning_lfa.py`

---

### Deep Q-Networks (DQN)

Deep Q-Networks (DQN) replace hand-crafted features with neural networks (NNs):

$$
Q(s,a;\theta) = \text{NN}_\theta(s)[a]
$$

where *the network outputs Q-values for all actions given a state*. **Note:** One can of course design the NN such that it takes a state-action pair as input and outputs the Q-value for that pair. However, this is not computation-efficient, as only one Q-value is generated in a forward pass.

While this significantly improves representational power, it introduces instability due to the combination of the three triads:
- **bootstrapping**
- **function approximation**  
- **off-policy learning**

To address these issues, DQN incorporates several key techniques:

#### Experience Replay
A replay buffer stores past transitions and samples mini-batches uniformly, **reducing temporal correlations and approximating i.i.d. sampling.**

#### Target Network
A separate network with frozen parameters is used to compute the target:

$$
y = r + \gamma \max_{a'} Q_{\theta^-}(s',a')
$$

This stabilizes training by **decoupling the target from the rapidly changing Q-network.**

#### Mini-batch Updates
Gradient-based optimization is performed over batches of transitions for **improved stability and efficiency.**

DQN can be viewed as a *nonlinear* extension of linear function approximation, where **the feature representation is learned jointly with the value function.**

**Implementation:** `algos/dqn.py`

Notably, we provide two implementations of DQN that differ in how data collection and training are scheduled.

- **Interleaved Training (Online Updates).**  
  In the first implementation, data collection and training are performed simultaneously. At each environment step, the agent stores the transition in the replay buffer and immediately performs a gradient update once the buffer size exceeds a threshold. This corresponds to a standard online DQN setup, where each interaction step is followed by a training step. While this approach is simple and sample-efficient, it may suffer from instability in early stages due to **highly correlated** samples. The implementation is provided in `dqn_interleave_buffer_and_training` :contentReference[oaicite:0]{index=0}.

- **Separated Training (Offline Updates).**  
  In the second implementation, we decouple data collection and training. The agent first interacts with the environment to populate the replay buffer, and then performs multiple gradient updates using the collected data. This approach is more aligned with batched or offline training paradigms, where updates are performed on a relatively stable dataset. Empirically, this can improve stability and allow for more controlled optimization. The implementation is provided in `dqn_separate_buffer_and_training` :contentReference[oaicite:1]{index=1}.

These two variants highlight an important design choice in deep RL: the balance between data collection and optimization. The interleaved version emphasizes responsiveness and continual learning, while the separated version emphasizes stability and better utilization of replayed experiences.