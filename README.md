# Implementation of Modern Reinforcement Learning (RL) Algorithms from Scratch

## Contents
- [Abstract](#abstract)
- [Environments](#environments)
  - [GridWorld](#gridworld)
  - [CartPole](#cartpole)
  - [LunarLander](#lunarlander)
- [Value-Based Algorithms](#value-based-algorithms)
  - [Tabular Q-Learning](#tabular-q-learning)
  - [Q-Learning with Linear Function Approximation (LFA)](#q-learning-with-linear-function-approximation-lfa)
  - [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
- [Policy Gradient Methods](#policy-gradient-methods)
  - [REINFORCE (Monte Carlo Policy Gradient)](#reinforce-monte-carlo-policy-gradient)
  - [A2C (Advantage Actor-Critic)](#a2c-advantage-actor-critic)


## Abstract
I am a PhD candidate in the *Department of Electrical and Computer Engineering* at North Carolina State University. My research focuses on the theory of reinforcement learning (RL), particularly on leveraging multi-agent structures to improve sample efficiency. Specifically, we develop algorithms with provable guarantees to demonstrate the benefits of collaboration and decentralized learning.

While much of our work is theoretical, this project aims to bridge theory and practice by implementing modern RL algorithms from scratch, including Q-learning, DQN, DDPG, PPO, etc. In addition, we explore extensions inspired by multi-agent RL to evaluate their empirical performance across different environments. This is an evolving project intended to provide both practical insights and theoretical intuition.

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
- **Reward function:** per-step survival reward from the base environment, picking discrete values in $0$ and $1$ based on the pole's angle and cart's position.
- **Stop criterion:** an episode ends on failure (terminated) or time-limit truncation.
- **Evaluation metric:** a return over 200 is considered a success, with the maximum return being 500.
This environment introduces:
- continuous state representation
- function approximation requirements for value estimation
- longer-horizon training dynamics relevant to DQN-style methods

It serves as a natural next step after GridWorld for evaluating linear approximation and neural value function learning.

---

### LunarLander

LunarLander is a more complex continuous-state control task where the agent must safely land a spacecraft on a designated pad.

- **State space:** an 8D continuous vector: lander position, velocity, angle, angular velocity, and leg contact indicators.
- **Action space:** four discrete actions: do nothing, fire left engine, fire main engine, fire right engine.
- **Transition model:** stochastic simulation dynamics from Gymnasium LunarLander-v3.
- **Reward function:** shaped reward based on distance to landing pad, velocity, angle, and leg contact, ranging from $-100$ to $+100$.
- **Stop criterion:** an episode ends on successful landing, crash, or time-limit truncation.
- **Evaluation metric:** a return over 200 is considered a success, with the maximum return being 300.

---

## Value-Based Algorithms

Value-based algorithms lie at the heart of reinforcement learning, focusing on learning estimates of the value (expected cumulative reward) for each state or state-action pair under a particular policy. By incrementally improving these estimates using observed experience, agents can derive effective control strategies through value optimization. In this section, we implement three representative methods, each building on the strengths and limitations of the previous:

- Tabular Q-learning  
- Q-learning with Linear Function Approximation (LFA)  
- Deep Q-Networks (DQN)  

Together, these methods trace the evolution of value-based RL from exact solutions in simple environments to scalable algorithms for complex, real-world tasks. By examining each approach in turn, we gain both theoretical insight and practical experience in the progression from tabular representations to modern deep RL.

---

### Tabular Q-Learning

Tabular Q-learning, originally proposed by Christopher Watkins, is one of the most fundamental algorithms in RL. It maintains an explicit table of Q-values for each state-action pair and updates them using stochastic approximations of the Bellman optimality operator.

Each update is given by:

$$
Q_{t+1}(s,a) = (1-\alpha) Q_t(s,a) + \alpha \left(r + \gamma \max_{a'} Q_t(s',a')\right)
$$

Under standard assumptions, tabular Q-learning converges to the optimal action-value function $Q^*$ with a rate of $\tilde{O}(1/\sqrt{T})$. Its simplicity and strong theoretical guarantees make it a natural starting point for understanding more advanced methods.

**Implementation:** `algos/Vanilla_Q.py`

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

**Implementation:** `algos/Q_LFA.py`

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

**Implementation:** `algos/DQN.py`

Notably, we provide two implementations of DQN that differ in how data collection and training are scheduled.

- **Interleaved Training (Online Updates).**  
  In this implementation, data collection and training are interleaved. At each environment step, the agent stores the transition in the replay buffer and performs a gradient update once the buffer size exceeds a threshold. This matches a standard *online* DQN setup, where each interaction step is followed by a training step. Because updates begin while the buffer is still being populated, the sampled transitions can be highly correlated. A common implementation detail is to **add a warmup period before training starts**. The implementation is provided in `algos、DQN.py/dqn_interleave_buffer_and_training`.

- **Separated Training (Offline Updates).**  
  In this implementation, data collection and training are separated. The agent first interacts with the environment to populate the replay buffer, and then performs multiple gradient updates using the collected data. This matches a batched training setup, where optimization is applied to a fixed dataset collected in advance. The implementation is provided in `algos/DQN.py/dqn_separate_buffer_and_training`.

These two variants highlight a design choice in deep RL: whether data collection and optimization happen in the same loop or in separate phases.

Notably, we would like to provide a careful discussion on the hyperparameters of DQN, such as learning rate, batch size, replay buffer size, target network update frequency, etc. These hyperparameters can significantly affect the performance and stability of DQN. For example, a large learning rate may lead to divergence, while a small learning rate may result in slow convergence. A large batch size can improve stability but may require more computational resources. A large replay buffer can provide more diverse experiences but may also introduce stale data. The target network update frequency can affect the bias-variance tradeoff in the target estimation.

- **Learning Rate:** A common choice is $1e-3$ or $1e-4$. A too large learning rat can cause divergence, while a too small learning rate can lead to slow convergence. It is often recommended to start with a moderate learning rate and adjust based on the observed training dynamics.

- **Batch Size:** A common choice is 32 or 64. A larger batch size can improve stability but may require more computational resources. A smaller batch size can lead to noisier updates but may allow for faster iterations.

- **Replay Buffer Size:** A common choice is 100,000 or 1,000,000 transitions. A larger replay buffer can provide more diverse experiences but may also introduce stale data. It is important to ensure that the buffer is large enough to capture a wide range of experiences while also being manageable in terms of memory. Particularly, if the buffer is too small, the agent may overfit to recent experiences, and fail to explore optimal policies.

- **Training Steps Per Episode:** This parameter is specific to the separated training implementation. A common choice is 100 or 200 training steps per episode (for a total of 1000 episodes). This allows the agent to perform multiple updates using the collected data, which can improve performance. As the number of training steps tends to infinity, this approach corresponds to the scheme of fitted Q-iteration (FQI), which can be computationally expensive and may lead to overfitting if the data is not sufficiently diverse. *Note*: the FQI is essentially applying the projected Bellman operator iteratively on a fixed dataset.

- **Target Network Update Frequency:** A target network is specifically designed to stabilize training by decoupling the target from the rapidly changing Q-network. A common choice is to update the target network every 1000 training steps. This can help to reduce the variance of the target estimates and improve stability. Particularly, if the target network is updated too frequently, it may not provide sufficient stability, while if it is updated too infrequently, it may lead to stale targets that do not reflect the current policy. Therefore, it is important to find a balance in the target network update frequency based on the observed training dynamics.

- **Epsilon Decay Schedule:** In an $\epsilon$-greedy exploration strategy, the choice of how $\epsilon$ decays over time can significantly affect the agent's ability to explore the environment effectively. A common approach is to start with a high $\epsilon$ (e.g., 1.0) and decay it gradually to a lower value (e.g., 0.01) over a certain number of episodes (e.g., 500 or 1000). This allows the agent to explore more in the early stages of training and exploit learned policies more as training progresses. This explorative parameter is ensured by the hyperparameter `epsilon_min', which sets the minimum value of $\epsilon$ after decay.

---

## Policy Gradient Methods

While value-based algorithms such as Q-learning and DQN focus on estimating value functions and deriving policies indirectly (via action selection from Q-values), **policy gradient methods** directly optimize the parameters of a policy by maximizing the expected sum of rewards. This direct approach is especially powerful in high-dimensional and continuous action spaces, or when differentiable policies are required.

### Key Motivation

- **Expressive Policies:** Policy gradients work with both deterministic and stochastic policies, improving exploration.
- **Continuous Actions:** They naturally handle continuous actions, where value-based methods often fail.
- **Direct Optimization:** They update policies in the direction that locally improves the expected return, using gradient ascent.

Despite these advantages, policy gradients often exhibit **high variance** and benefit from techniques like baselines or actor-critic architectures for better learning efficiency.

---

### REINFORCE (Monte Carlo Policy Gradient)

**REINFORCE** (Williams, 1992) is a foundational Monte Carlo policy gradient algorithm. It uses complete episode returns to update policy parameters based on performance.

#### Objective

Given a parameterized stochastic policy $\pi_\theta(a \mid s)$, the objective is to maximize the expected return:

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t r_t \right]
```

The policy gradient theorem tells us:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, G_t \right]
```

where $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} r_k$ is the discounted return from time $t$ onward.

---

#### Algorithm (Episodic)

1. **Initialize** policy parameters $\theta$
2. **Repeat:**
    - Generate an **episode** $\tau = (s_0, a_0, r_0, ..., s_T)$ by sampling actions from $\pi_\theta$
    - For each step $t$ in $\tau$:
        - Compute return $G_t = \sum_{t'=t}^T \gamma^{t'-t} r_{t'}$
        - Update parameters:
          
```math
\theta \leftarrow \theta + \eta \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t
```

3. **Until** convergence

---

#### Practical Notes

- The policy is typically represented by a neural network outputting *action probabilities* (for discrete actions) or *distribution parameters* (for continuous actions).
- Log-probabilities and rewards are stored for each episode, and the policy is updated once at the end of each trajectory. **Note**: the code for implementation is DIFFERENT from the pseudocode; specifically, in the pseudocode, the policy is updated for $\tau$ steps, which is valid since the gradients are evaluated at the same parameters $\theta$. In practice, we often perform a single update after processing the entire episode, since if we update the policy multiple times within the same episode, the gradients will be evaluated at different parameters, which may not correspond to the original policy that generated the trajectory. This can lead to biased updates and unstable learning dynamics.
- Variance reduction is important: subtracting a baseline (e.g. an average return) from $G_t$ does not introduce bias but reduces variance significantly.

The REINFORCE algorithm is implemented in `algos/REINFORCE.py` and serves as a starting point for understanding policy gradient methods. It provides a clear illustration of how to directly optimize policies using sampled returns, while also highlighting the challenges of high variance and the need for further improvements.

---

### A2C (Advantage Actor-Critic)

The REINFORCE algorithm suffers from **high variance** in policy updates due to the use of complete episode returns, making training unstable. **A2C** is an **actor-critic** method that improves upon vanilla REINFORCE by introducing a learned **critic** as a baseline to reduce gradient variance.

- **Actor:** a parameterized policy $\pi_\theta(a \mid s)$ that outputs a distribution over actions.
- **Critic:** a parameterized value function $V_\phi(s)$ that estimates the expected return from state $s$.

The actor is updated using an **advantage estimate**, which measures how good an action is relative to the baseline value $V_\phi(s)$.

---

#### Advantage (1-step TD)

A commonly used advantage estimator in A2C is the **1-step temporal-difference (TD) advantage**:
```math
\hat{A}_t = r_t + \gamma \, V_\phi(s_{t+1}) - V_\phi(s_t)
```

Equivalently, define the TD target
```math
y_t = r_t + \gamma \, V_\phi(s_{t+1})
```
and compute
```math
\hat{A}_t = y_t - V_\phi(s_t)
```

---

#### Loss Functions

Theoretically, the A2C algorithm could have separate neural networks for the actor and critic, but in practice, it is common to use a shared backbone with two heads (one for the policy and one for the value function). The loss functions for the actor and critic are defined as follows:

**Actor loss** (maximize expected return; implemented as minimizing the negative objective):

The policy gradient theorem states that the gradient of the objective can be written as:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a \sim \pi_\theta}
\left[ A^\pi(s,a) \nabla_\theta \log \pi_\theta(a \mid s)
\right]
$$

In practice, this expectation is approximated using samples, yielding the following loss function for the actor:
```math
\mathcal{L}_{\text{actor}}(\theta) = -\log \pi_\theta(a_t \mid s_t)\, \hat{A}_t
```

**Critic loss** (value regression via TD error):
```math
\mathcal{L}_{\text{critic}}(\phi) = \hat{A}_t^2
```
Other critic loss formulations include the Huber loss or the L1 loss, which can be more robust to outliers in the TD error. To be specific, the Huber loss is defined as:
```math
\mathcal{L}_{\text{critic}}(\phi) = \begin{cases}
\frac{1}{2} \hat{A}_t^2 & \text{if } |\hat{A}_t| \leq \delta \\
\delta (|\hat{A}_t| - \frac{1}{2} \delta) & \text{otherwise}
\end{cases}
```
where $\delta$ is a hyperparameter that controls the transition point between the quadratic and linear regions of the loss.

An additional **entropy bonus** can be included to encourage exploration:
```math
\mathcal{L}_{\text{entropy}}(\theta) = -\mathcal{H}\bigl(\pi_\theta(\cdot \mid s_t)\bigr)
```

**Total loss** (typical weighting):
```math
\mathcal{L} = \mathcal{L}_{\text{actor}} + 0.5\,\mathcal{L}_{\text{critic}} + 0.01\,\mathcal{L}_{\text{entropy}}
```

In practice, the advantage term is usually **detached** when computing the actor loss so that the actor does not backpropagate through the critic.

---

#### Algorithm

1. **Initialize** actor parameters $\theta$ and critic parameters $\phi$

2. **Repeat:**
   - Generate an episode $\tau = (s_0, a_0, r_0, \ldots, s_T)$ by sampling actions from $\pi_\theta$
   - For each step $t$ in $\tau$:
     - Compute TD target: $y_t = r_t + \gamma \, V_\phi(s_{t+1})$
     - Compute advantage: $\hat A_t = y_t - V_\phi(s_t)$
     - Update actor & critic parameters:

$$
\theta \leftarrow \theta + \eta_\theta \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat A_t
$$

$$
\phi \leftarrow \phi - \eta_\phi \nabla_\phi \left(y_t - V_\phi(s_t)\right)^2
$$

3. **Until** convergence

---

#### Issues with 1-step TD Advantage:
1. **Bias:** The 1-step TD target is a biased estimate of the true return, especially when the value function used to compute the advantage is inaccurate. This can lead to suboptimal policy updates and slow learning.
2. **High Variance:** There are a few reasons for high variance in the 1-step TD advantage:
   - Only one sample of the advantage is used per update, which can lead to high variance in the gradient estimates.
   - There is temporal correlation between the advantage estimates across time steps, which can further increase variance.
   - The critic's value estimates can be noisy, especially early in training, which can lead to high variance in the advantage estimates.

---

#### Potential Solutions:
##### Multi-step Returns
To reduce bias, we can use multi-step returns instead of 1-step TD targets.  
An $n$-step return starting at time $t$ is defined as:

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V_\phi(s_{t+n})
$$

Given a rollout of length $n$, we can compute an $n$-step return for each time step within the rollout.  
For $k = 0, \dots, n-1$:

$$
G_{t+k}^{(n-k)} = \sum_{l=0}^{n-k-1} \gamma^l r_{t+k+l} + \gamma^{n-k} V_\phi(s_{t+n})
$$

The corresponding advantage estimates are:

$$
\hat A_{t+k} = G_{t+k}^{(n-k)} - V_\phi(s_{t+k}), \quad k = 0, \dots, n-1
$$

We then compute the policy gradient using these per-step advantages and average across the rollout:

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{n} \sum_{k=0}^{n-1}
\hat A_{t+k} \nabla_\theta \log \pi_\theta(a_{t+k} \mid s_{t+k})
$$

This approach can significantly reduce bias compared to 1-step TD targets, as it incorporates more actual rewards from the environment. However, it may still suffer from high variance, especially if the rollout length $n$ is large. As a result, we introduce the generalized advantage estimation (GAE), which provides a more flexible way to balance bias and variance by weighting returns of different lengths.

##### GAE (Generalized Advantage Estimation)
The GAE is a method that combines multi-step returns with an *exponentially weighted average* to provide a more stable advantage estimate. The GAE advantage is defined as:

$$\hat A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ is the TD error at time $t$. The hyperparameter $\lambda \in [0,1]$ controls the bias-variance tradeoff: $\lambda=0$ corresponds to 1-step TD (high bias), while $\lambda=1$ corresponds to using the full return (high variance). By tuning $\lambda$, we can achieve a more stable learning process with reduced variance compared to using multi-step returns alone. 

In practice, we use the truncated version of GAE, where we only sum up to a finite horizon $n$:

$$\hat A_{t+k}^{\text{GAE}(\gamma, \lambda, n)} = \sum_{l=0}^{n-k-1} (\gamma \lambda)^l \delta_{t+l}$$

The rest of the process is similar to the multi-step return approach, where we compute the policy gradients using these GAE advantages and average over the rollout.

GAE is more stable because it:

- reduces long-horizon reward noise  
- exponentially discounts unreliable future estimates  
- smooths advantage signals over time  
- provides a tunable bias–variance tradeoff

##### Parallel Environments
To further reduce variance and improve sample efficiency, we can run multiple **parallel environments** to collect more diverse experiences in each update. This allows us to compute policy gradients using a larger batch of data, which can lead to more stable updates and faster convergence. In practice, we can use vectorized environments (e.g., `gym.vector`) to manage multiple instances of the environment simultaneously, allowing for efficient data collection and processing.

The final version for A2C with GAE and parallel environments is implemented in `algos/A2C.py/a2c_multi_env()`. This implementation incorporates all the techniques discussed above to provide a more robust and efficient policy gradient method compared to vanilla REINFORCE.




