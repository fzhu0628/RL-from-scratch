\usepackage{amsmath}
\section{Implementation of Modern Reinforcement Learning Algorithms from Scratch}

\textbf{Abstract.}
I am a Ph.D. candidate in the \textit{Department of Electrical and Computer Engineering} at \textit{North Carolina State University}. My research focuses on the theory of reinforcement learning (RL), particularly on leveraging multi-agent structures to improve sample efficiency. Specifically, we develop algorithms with provable guarantees to demonstrate the benefits of collaboration and decentralized learning.

While much of our work is theoretical, this project aims to bridge theory and practice by implementing modern RL algorithms from scratch, including Q-learning, DQN, DDPG, TRPO, and PPO. In addition, we explore extensions inspired by our multi-agent research to evaluate their empirical performance across different environments. This is an evolving project intended to provide both practical insights and theoretical intuition.

\subsection{Value-Based Algorithms}

In this section, we implement three representative value-based methods: tabular Q-learning, Q-learning with linear function approximation (LFA), and deep Q-learning (DQN). These methods illustrate the progression from exact representations to function approximation and deep learning.

\subsubsection{Tabular Q-Learning}

Tabular Q-learning, originally proposed by Watkins, is one of the most fundamental algorithms in reinforcement learning. It maintains an explicit table of Q-values for each state-action pair and updates them using stochastic approximations of the Bellman optimality operator.

Each update can be written as:
\begin{equation}
Q_{t+1}(s,a) = (1-\alpha) Q_t(s,a) + \alpha \left(r + \gamma \max_{a'} Q_t(s',a')\right).
\end{equation}

Under standard assumptions, tabular Q-learning converges to the optimal action-value function $Q^*$ with a rate of $\tilde{O}(1/\sqrt{T})$. Its simplicity and strong theoretical guarantees make it a natural starting point for understanding more advanced methods.

The implementation can be found in \texttt{algos/q\_learning.py}.

\subsubsection{Q-Learning with Linear Function Approximation (LFA)}

To handle larger or continuous state spaces, tabular representations become infeasible. A natural extension is to approximate the Q-function using a linear model:
\begin{equation}
Q(s,a; \theta) = \theta^\top \phi(s,a),
\end{equation}
where $\phi(s,a)$ is a feature representation of the state-action pair.

This approach enables generalization across states while significantly reducing memory requirements. However, unlike the tabular setting, the parameters $\theta$ are shared across all state-action pairs. As a result, updates are no longer local, and the algorithm effectively performs a projected Bellman update:
\begin{equation}
Q \approx \Pi T Q,
\end{equation}
where $\Pi$ denotes projection onto the function class.

This introduces several challenges, including the loss of contraction properties, potential instability, and sensitivity to feature design. In our implementation, we explore simple feature mappings such as normalized state coordinates and one-hot encodings of actions.

The implementation is provided in \texttt{algos/q\_learning\_lfa.py}.

\subsubsection{Deep Q-Learning (DQN)}

Deep Q-Networks (DQN) replace hand-crafted features with neural networks:
\begin{equation}
Q(s,a;\theta) = \text{NN}_\theta(s)[a],
\end{equation}
where the network outputs Q-values for all actions given a state.

While this significantly improves representational power, it introduces instability due to the combination of bootstrapping, function approximation, and off-policy learning. To address these issues, DQN incorporates several key techniques:

\begin{itemize}
    \item \textbf{Experience Replay:} A replay buffer stores past transitions and samples mini-batches uniformly, reducing temporal correlations and approximating i.i.d. sampling.
    
    \item \textbf{Target Network:} A separate network with frozen parameters is used to compute the target:
    \begin{equation}
    y = r + \gamma \max_{a'} Q_{\theta^-}(s',a'),
    \end{equation}
    which stabilizes training by decoupling the target from the rapidly changing Q-network.
    
    \item \textbf{Mini-batch Updates:} Gradient-based optimization is performed over batches of transitions for improved stability and efficiency.
\end{itemize}

DQN can be viewed as a nonlinear extension of linear function approximation, where the feature representation is learned jointly with the value function.

The implementation is available in \texttt{algos/dqn.py}.