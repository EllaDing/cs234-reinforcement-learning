# Learning Notes - Lecture 6 part1
Value-based, policy-based, and actor-critic methods are three main families in reinforcement learning that differ in what they learn and how they derive the agent's behavior. In previous lectures, we learned about policy search method. In lecture 6, we will focus on actor-critic methods.

## Actor-Critic Framework
- **Actor**: The policy (e.g., a neural network) that selects actions. It is parameterized by $ \theta $ and aims to maximize the expected cumulative reward.
- **Critic**: A value function (e.g., $ V(s) $ or $ Q(s,a) $) that evaluates the quality of the actor’s actions. It is parameterized by $ \phi $ and estimates the expected return from states or state-action pairs.

They merge the policy approach and value approach:
  - The **actor** is the policy that selects actions.
  - The **critic** is a value function estimator that evaluates how good the current state (or state-action pair) is.
### Key Interaction
1. The **actor** selects an action $ a \sim \pi_\theta(a|s) $ in state $ s $.
2. The **critic** evaluates the action by computing a value estimate (e.g., $ V(s) $).
3. The actor updates its policy using feedback from the critic, while the critic improves its estimates based on observed rewards.

### Relationship Between Policy-Based + Baseline and Actor-Critic
The baseline method reduces variance in policy gradient updates by subtracting a state-dependent term $ b(s) $ from the return $ R $:
$$
\nabla_\theta J(\theta) \approx \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left(R(\tau) - b(s_t)\right) \right].
$$

This formulation is essentially the core of actor-critic algorithms:
- **In a pure policy gradient method (like REINFORCE),** a baseline may be added to reduce variance.  
- **In actor-critic methods,** the critic is not just a fixed baseline but is a learned, adaptive function that continuously provides feedback to the actor. This joint training leads to more efficient updates.

In actor-critic algorithms, the **critic directly provides the baseline** through its value function estimates. Specifically:
- **Critic as Baseline**: The critic estimates $ V(s) $, which serves as the baseline $ b(s) $.
- **Advantage Function**: The actor uses the **advantage** $ A(s,a) = Q(s,a) - V(s) $, where:
  - $ Q(s,a) $: Expected return after taking action $ a $ in state $ s $.
  - $ V(s) $: Expected return from state $ s $ under the current policy.

#### Actor Update
$$
\nabla_\theta J(\theta) \approx \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) \right].
$$
- The advantage $ A(s,a) $ replaces $ R(\tau) - b(s) $, providing a **relative measure** of action quality.

---

### Why Actor-Critic Reduces Variance
1. **Adaptive Baseline**: Unlike static baselines (e.g., average reward), $ V(s) $ is learned and adapts to the policy’s current performance, offering a more precise reference for action quality.
2. **Temporal Difference (TD) Learning**: The critic updates $ V(s) $ using TD error $ \delta = r + \gamma V(s') - V(s) $, which provides low-variance, incremental updates. Note: critic can use any combo of MC and TD estimator through multi-step estimator.
3. **Focus on Advantage**: By evaluating actions relative to the state average $ V(s) $, the actor prioritizes actions that outperform the policy’s average behavior, filtering out irrelevant noise.

---

### Example: Advantage Actor-Critic (A2C)
- **Critic**: Learns $ V_\phi(s) $ to approximate the state-value function.
- **Actor**: Updates $ \pi_\theta(a|s) $ using the advantage $ A(s,a) = Q(s,a) - V(s) $.
- **Algorithm**:
  1. Collect trajectory $ \tau = (s_0, a_0, r_0, ..., s_T) $.
  2. Compute TD targets for the critic: $ V_{\text{target}}(s_t) = r_t + \gamma V_\phi(s_{t+1}) $.
  3. Update critic $ \phi $ to minimize $ \|V_\phi(s_t) - V_{\text{target}}(s_t)\|^2 $.
  4. Update actor $ \theta $ using $ \nabla_\theta J(\theta) \propto \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) $.

---

### Benefits Over Pure Policy Gradients
- **Lower Variance**: The critic’s baseline $ V(s) $ provides a stable reference, reducing the noise in gradient estimates.
- **Faster Convergence**: By leveraging value function approximations, actor-critic methods often learn more efficiently than Monte Carlo-based policy gradients.
- **Adaptability**: The critic continuously refines its baseline as the policy improves, enabling dynamic credit assignment.

---

### N-step estimator
N-step estimator in actor-critic methods combines the strengths of Monte Carlo (MC) and Temporal Difference (TD) estimators to balance bias and variance.

It computes the target value for a state $s_t$ by summing $N$ immediate rewards and bootstrapping the remaining return using the critic’s value estimate $V(s_{t+N})$:
 $$
  R_t^{(N)} = \sum_{k=0}^{N-1} \gamma^k r_{t+k} + \gamma^N V(s_{t+N}).
 $$
  - **MC Component**: Uses $N$-step actual rewards $\sum_{k=0}^{N-1} \gamma^k r_{t+k}$.
  - **TD Component**: Bootstraps with $\gamma^N V(s_{t+N})$.

#### Bias-Variance Tradeoff
- **Small $N$** (e.g., TD(0)): Lower variance (fewer rewards) but higher bias (relies on critic’s estimate).
- **Large $N$** (e.g., MC): Lower bias (more rewards) but higher variance (longer dependency chain).

#### Actor-Critic Integration
- The critic estimates $V(s)$, which is used to compute $R_t^{(N)}$.
- The actor updates its policy using the advantage $A(s_t, a_t) = R_t^{(N)} - V(s_t)$.

---

### Choosing the Optimal $N$: Minimizing MSE
The optimal $N$ minimizes the **Mean Squared Error (MSE)** between the N-step target $R_t^{(N)}$ and the true value $V^\pi(s_t)$:
$$
\text{MSE}(N) = \mathbb{E}\left[ \left( R_t^{(N)} - V^\pi(s_t) \right)^2 \right].
$$

#### MSE Decomposition
$$
\text{MSE}(N) = \underbrace{\text{Bias}(N)^2}_{\text{From TD bootstrapping}} + \underbrace{\text{Variance}(N)}_{\text{From MC rewards}}.
$$
- **Bias**: Increases as $N \downarrow$ (more reliance on critic).
- **Variance**: Increases as $N \uparrow$ (more stochastic rewards).

#### Optimal $N$
The optimal $N$ is where the **sum of bias² and variance is minimized**:
$$
N^* = \arg\min_N \left( \text{Bias}(N)^2 + \text{Variance}(N) \right).
$$

Why using MSE:
- **MSE as a Proxy for Accuracy**: A low MSE ensures the critic’s estimates $V(s)$ are close to the true values $V^\pi(s)$, improving policy updates.
- **Bias-Variance Tradeoff**: MSE explicitly balances the two sources of error, unlike metrics like reward maximization, which are confounded by exploration.
---

### Example: N-Step Advantage Actor-Critic (A2C)
1. **Critic Update**:
  $$
   V_\phi(s_t) \leftarrow V_\phi(s_t) + \alpha \left( R_t^{(N)} - V_\phi(s_t) \right),
  $$
   where $R_t^{(N)} = \sum_{k=0}^{N-1} \gamma^k r_{t+k} + \gamma^N V_\phi(s_{t+N})$.

2. **Actor Update**:
  $$
   \nabla_\theta J(\theta) \propto \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left( R_t^{(N)} - V_\phi(s_t) \right).
  $$
