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
2. **Temporal Difference (TD) Learning**: The critic updates $ V(s) $ using TD error $ \delta = r + \gamma V(s') - V(s) $, which provides low-variance, incremental updates. Note: critic can use any combo of MC and TD estimator through multi-step estimator (e.g. first n-step using MC estimator and the rest using bootstrap).
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