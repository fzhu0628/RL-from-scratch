# Implementation of Modern Reinforcement Learning (RL) Algorithms from Scratch

## Contents
- [Abstract](#abstract)
- [Environments](#environments)
  - [GridWorld](#gridworld)
  - [CartPole](#cartpole)
- [Value-Based Algorithms](#value-based-algorithms)
  - [Tabular Q-Learning](#tabular-q-learning)
  - [Q-Learning with Linear Function Approximation (LFA)](#q-learning-with-linear-function-approximation-lfa)
  - [Deep Q-Networks (DQN)](#deep-q-networks-dqn)


## Abstract
I am a PhD candidate in the *Department of Electrical and Computer Engineering* at North Carolina State University. My research focuses on the theory of reinforcement learning (RL), particularly on leveraging multi-agent structures to improve sample efficiency. Specifically, we develop algorithms with provable guarantees to demonstrate the benefits of collaboration and decentralized learning.

While much of our work is theoretical, this project aims to bridge theory and practice by implementing modern RL algorithms from scratch, including Q-learning, DQN, DDPG, TRPO, PPO, etc. In addition, we explore extensions inspired by multi-agent RL to evaluate their empirical performance across different environments. This is an evolving project intended to provide both practical insights and theoretical intuition.

---

## Environments

We evaluate our implementations on a set of environments with increasing complexity, starting from simple tabular settings and progressing to continuous-state control problems.

### GridWorld

GridWorld is a deterministic navigation task implemented on a fixed 2D lattice.

- **State space:** discrete coordinates $(x,y)$ on a $16 \times 16$ grid, with start state $(0,0)$ and goal state $(15,15)$.
- **Action space:** four discrete actions: up, down, left, right.
- **Transition model:** deterministic movement; attempts to move outside the grid are clipped at boundaries.
- **Reward function:** $-5$ for each non-goal step and $0$ at the goal.
- **Stop criterion:** an episode ends when the agent reaches the goal state.

This environment is useful for:
- validating tabular Bellman updates and policy extraction
- checking convergence behavior in a controlled MDP
- debugging exploration and reward-shaping effects

Because the state is low-dimensional and fully observable, GridWorld is mainly a correctness and diagnostics benchmark rather than a strong test of deep function approximation.

---

### CartPole

CartPole is a continuous-state control task where the agent applies horizontal force to keep an inverted pole balanced.

- **State space:** a 4D continuous vector: cart position, cart velocity, pole angle, and pole angular velocity.
- **Action space:** two discrete actions: push left or push right.
- **Transition model:** stochastic simulation dynamics from Gymnasium CartPole-v1.
- **Reward function:** per-step survival reward from the base environment.
- **Stop criterion:** an episode ends on failure (terminated) or time-limit truncation.

This environment introduces:
- continuous state representation
- function approximation requirements for value estimation
- longer-horizon training dynamics relevant to DQN-style methods

It serves as a natural next step after GridWorld for evaluating linear approximation and neural value function learning.

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

While this significantly improves representational power, it introduces instability due to the combination of the *three triads*:
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
  In this implementation, data collection and training are interleaved. At each environment step, the agent stores the transition in the replay buffer and performs a gradient update once the buffer size exceeds a threshold. This matches a standard *online* DQN setup, where each interaction step is followed by a training step. Because updates begin while the buffer is still being populated, the sampled transitions can be highly correlated. A common implementation detail is to **add a warmup period before training starts**. The implementation is provided in `algos/dqn_interleave_buffer_and_training`.

- **Separated Training (Offline Updates).**  
  In this implementation, data collection and training are separated. The agent first interacts with the environment to populate the replay buffer, and then performs multiple gradient updates using the collected data. This matches a batched training setup, where optimization is applied to a fixed dataset collected in advance. The implementation is provided in `algos/dqn_separate_buffer_and_training`.

These two variants highlight a design choice in deep RL: whether data collection and optimization happen in the same loop or in separate phases.