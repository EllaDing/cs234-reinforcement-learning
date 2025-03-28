# Problem of Policy Gradients
- **Sample Efficiency is Poor**: need many rollouts with the same policy to get good estimates of the gradient.
- **Distance in the Parameter Space Doesn't Equal to the Policy Space**: The idea is if you have a gradient optimization. When you change your parameter, does it smoothly chance how the policy value change? We want to know wheter it is smooth or not. A small change in the parameter space a little bit may change the action a lot if these 2 space are not proportional.

## Sample Efficiency - How to Use Old Data
Currently we are using on-policy approach, where policy is updated and then used in the new data sampling. Ideally we want to reuse the old data to estimate new gradients but without introducing much unstability.

Key Challange: estimate the performance of new policy using data sampled from old policy.

## Smooth Optimization
Non-Stationary Gradients: In RL, gradients vary widely in magnitude due to sparse rewards, changing policies, and high-variance advantage estimates.

### 1. How to Choose Step Size
$$
\theta _{k+1} = \theta _k + \alpha_{k}\hat g_k
$$
with step $\nabla_k=\alpha \hat g_k$

Automatic learning rate adjustment like advantage normalization, or Adam-style optimizers can help.

**Adam**: Combines momentum (exponentially decaying gradient history) and adaptive per-parameter learning rates.
   - Update rule:
     $$
     \theta_{k+1} = \theta_k - \alpha \cdot \frac{m_k}{\sqrt{v_k} + \epsilon},
     $$
     where $ m_k $ (biased first moment) and $ v_k $ (biased second moment) are estimates of the gradient mean and variance.
   - **Why it works**: Scales learning rates by gradient magnitudes, dampening updates for parameters with large gradients.

### Policy Performance Bound
  Use rollouts collected from the most recent policy as efficiently as possible and respect distance in policy space

Benefits:
- Improve sample efficiency.
- Keep Policy Update as smooth as possible (not only consider the parameter space).

#### How to Estimate $\pi'$ Given Data from $\pi$
Calculate the difference between values of 2 policies using samples from $\pi$.

$$
J(\pi') - J(\pi) = E_{trajectories \sim \pi'}[\sum_{t=0}^{\infin}\gamma^t A^{\pi}(s_t, a_t)]
$$

Grouping the timestep of different trajectories by states and actions:

$$
J(\pi') - J(\pi) = \frac{1}{1-\gamma} E_{s \sim d^{\pi'}, a \sim \pi'}[A^\pi(s, a)]
$$

See details of $d^\pi(s)$ in appendix 1.

Ideally we can maximize the difference to find the best $\pi'$ using samples from $\pi$. However, given that the equation still contains states and actions from $\pi'$, we still need to do some approximation:
1. importance sampling based on $V(s) = \sum_a \pi(a | s)Q(a, s)$. This helps swap action from policy $\pi'$ to $\pi$. But it cannot deal with the state distribution
2. If KL divergence of $\pi$ and $\pi'$ is bounded, we can approximate $s \sim \pi$ as $s \sim \pi'$.

Then the problem becomes a constrained optimization problem.
### Algorithms
Proximal Policy Optimization (PPO) leverages **KL divergence** (explicitly via penalties or implicitly via clipping) to ensure policy updates are stable, while the **value function** provides accurate advantage estimates. By balancing gradual policy changes with value function refinement, PPO achieves robust and sample-efficient learning.

**Policy Optimization Objective**

PPO aims to maximize the expected return while ensuring the new policy ($\pi_{\text{new}}$) does not deviate too far from the old policy ($\pi_{\text{old}}$). 

#### **KL Penalty**:
 Adds a KL divergence term to the objective:
     $$
     \mathcal{L}^{\text{KL}} = \mathbb{E} \left[ r(\theta) \hat{A}(s,a) - \beta D_{\text{KL}}(\pi_{\text{old}} \parallel \pi_{\text{new}}) \right].
     $$
Adjusts $\beta$ dynamically: Increase if KL exceeds threshold, decrease otherwise.

$r(\theta)$ directly measures how much the policy has changed, decoupling policy optimization from reward scales. It doesn't have a relationship with reward function $R(s, a)$. Here we focus on the difference between a new policy and old policy, instead of the reward from the environment.

**Advantage Function Definition**:
- The advantage function $A(s,a)$ measures how much better an action $a$ is compared to the average action in state $s$ under the policy:
$$
A(s,a) = Q(s,a) - V(s),
$$

#### **Clipping (Default)**:
- Uses hard constraints on $r(\theta)$, avoiding explicit KL computation.
- Indirectly bounds KL divergence by limiting policy ratio changes.

Purpose of $r(\theta)$:
- Quantifies how much the new policy deviates from the old policy for a given action $a$ in state $s$.
- Directly measures the relative likelihood of actions under the new vs. old policy.

Objective function:
$$
\mathcal{L}^{\text{CLIP}} = \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[ \min\left( r(\theta) \hat{A}(s,a), \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon) \hat{A}(s,a) \right) \right],
$$
where:
- $ r(\theta) = \frac{\pi_{\text{new}}(a|s)}{\pi_{\text{old}}(a|s)} $: It is the probability ratio of actions under the new vs. old policy. It came from **importance sampling**: When updating the policy, we reuse trajectories collected under the old policy. The ratio $r(\theta)$ reweights these trajectories to approximate expectations under the new policy:
     $$
     \mathbb{E}_{s,a \sim \pi_{\text{old}}} \left[ r(\theta) \hat{A}(s,a) \right] \approx \mathbb{E}_{s,a \sim \pi_{\text{new}}} \left[ \hat{A}(s,a) \right].
     $$
- $ \hat{A}(s,a) $: Advantage function estimating the relative value of action $a$ in state $s$.
- **Clipping** ($ \epsilon $): Limits $ r(\theta) $ to $[1-\epsilon, 1+\epsilon]$, indirectly controlling policy divergence.

#### **Value Function Differences**
- **Critic Network**: Estimates $ V(s) $, the expected return from state $s$.
- **Advantage Calculation**: 
  $$
  \hat{A}(s,a) = R(s,a) + \gamma V_{\text{old}}(s') - V_{\text{old}}(s),
  $$
  where $ V_{\text{old}} $ is the value function under $\pi_{\text{old}}$.

The advantage function estimate $\hat{A}(s,a) = R(s,a) + \gamma V_{\text{old}}(s') - V_{\text{old}}(s)$ in Proximal Policy Optimization (PPO) is derived as follows:

Context:
- In temporal difference (TD) learning, the Q-value can be approximated using a one-step lookahead:
$$
Q(s,a) \approx R(s,a) + \gamma V(s'),
$$
- Using the one-step Q-value approximation:
$$
A(s,a) \approx R(s,a) + \gamma V(s') - V(s).
$$
- In PPO, trajectories are collected using the **old policy** $\pi_{\text{old}}$. To ensure stable updates, the value function $V_{\text{old}}$ (from $\pi_{\text{old}}$) is used instead of the current value function. This avoids variance from using a changing value function during policy updates.
- Thus, the advantage estimate becomes:
$$
\hat{A}(s,a) = R(s,a) + \gamma V_{\text{old}}(s') - V_{\text{old}}(s).
$$

#### **Value Function Updates**
- **Objective**: Minimize Mean Squared Error (MSE) between predicted and observed returns:
  $$
  \mathcal{L}^{\text{VF}} = \mathbb{E} \left[ \left( V_{\text{new}}(s) - V_{\text{target}}(s) \right)^2 \right].
  $$
- **Key Point**: The new value function $ V_{\text{new}} $ adapts to $\pi_{\text{new}}$, but policy clipping ensures gradual shifts in $ V $-estimates.

#### Appendix 1 - Discounted State Distribution under Policy $\pi$
$d^\pi(s)$ represents the expected discounted visitation frequency of states when following policy $\pi$. It accounts for the discount factor $\gamma \in [0, 1)$, which de-emphasizes future states.

The discounted state distribution is defined as:
$$
d^\pi(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s \mid \pi),
$$
where:
- $P(s_t = s \mid \pi)$: Probability of being in state $s$ at time $t$ under policy $\pi$.
- $\gamma^t$: Discount factor applied to future states.
- $(1 - \gamma)$: Normalization term ensuring $\sum_{s} d^\pi(s) = 1$.

#### Appendix 2 - KL Divergence
For a given state $s$, the KL divergence between $\pi_1$ and $\pi_2$ is:
$$
D_{\text{KL}}(\pi_1 \parallel \pi_2)(s) = \mathbb{E}_{a \sim \pi_1(\cdot|s)} \left[ \log \frac{\pi_1(a|s)}{\pi_2(a|s)} \right].
$$
This quantifies how much $\pi_2$ diverges from $\pi_1$ in state $s$. To compute the **average divergence** over all states visited by $\pi_1$, we take the expectation over $\pi_1$’s state distribution $d^{\pi_1}(s)$:
$$
D_{\text{KL}}^{\text{avg}}(\pi_1 \parallel \pi_2) = \mathbb{E}_{s \sim d^{\pi_1}} \left[ D_{\text{KL}}(\pi_1 \parallel \pi_2)(s) \right].
$$

##### Practical Calculation Using Data from $\pi_1$
When using trajectories sampled from $\pi_1$:
1. **Collect Data**: Generate $N$ trajectories using $\pi_1$, yielding state-action pairs $\{(s_i, a_i)\}_{i=1}^M$.
2. **Compute Log-Probabilities**: For each $(s_i, a_i)$, calculate:
   - $\log \pi_1(a_i|s_i)$ (action likelihood under $\pi_1$),
   - $\log \pi_2(a_i|s_i)$ (action likelihood under $\pi_2$).
3. **Estimate KL Divergence**:
   $$
   \widehat{D}_{\text{KL}} = \frac{1}{M} \sum_{i=1}^M \left[ \log \pi_1(a_i|s_i) - \log \pi_2(a_i|s_i) \right].
   $$
   This is a **Monte Carlo estimate** of $D_{\text{KL}}^{\text{avg}}(\pi_1 \parallel \pi_2)$.

##### Key Considerations
**a. Why Use $\pi_1$’s Data?**
- The KL divergence is evaluated over states $s \sim d^{\pi_1}(s)$, as $\pi_2$ might visit different states. Using $\pi_1$’s data ensures we measure divergence in regions relevant to $\pi_1$.
- This avoids needing to sample from $\pi_2$, which could be computationally expensive or unsafe.

#### Appendix 3 - GAE: Generalized Advantage Estimator
**Generalized Advantage Estimator (GAE)** offers several advantages over using a fixed **N-step estimator** by combining multiple k-step estimators through an exponentially weighted average.
- **N-step Estimator**: Fixes a single horizon $N$ to combine MC value estimator and TD value estimator.
- **GAE**: Blends all k-step estimators (from $k=0$ to $k=T$) using a decay factor $\lambda$:
  $$
  \hat{A}_{\text{GAE}} = \sum_{k=0}^{T} (\gamma \lambda)^k \delta_{t+k},
  $$
  where $\delta_{t} = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.
  - **Exponential Decay**: Weights decay as $(\gamma \lambda)^k$, emphasizing shorter-term rewards (lower variance) while still accounting for long-term effects (lower bias).
  - **Tunable $\lambda$**:
    - $\lambda \to 0$: Reduces to TD(0) (low variance, high bias).
    - $\lambda \to 1$: Approaches Monte Carlo (low bias, high variance).

Benefit:
- **Single Hyperparameter ($\lambda$)**: Adjusting $\lambda$ provides a smooth interpolation between TD(0) and Monte Carlo, simplifying hyperparameter search.
- **Eligibility Traces**: GAE generalizes TD($\lambda$), a well-studied method for balancing bias and variance.
- **Principled Weighting**: The $(\gamma \lambda)^k$ decay ensures theoretical consistency with value function approximation.
- Automatically adapts weighting based on $\lambda$, performing well across diverse tasks (e.g., sparse/dense rewards).

Tradeoffs for $\lambda$ tuning:
- **Short-Term Rewards**: Lower $k$-steps (weighted more for $\lambda < 1$) provide stable, immediate feedback.
- **Long-Term Rewards**: Higher $k$-steps (with decaying weights) mitigate myopia without introducing excessive variance.

While GAE seems complex, it can be computed efficiently in $O(T)$ time using dynamic programming:
  $$
  \hat{A}_t = \delta_t + \gamma \lambda \hat{A}_{t+1}.
  $$
- **Vectorization**: Modern frameworks (e.g., PyTorch, TensorFlow) parallelize these computations.
