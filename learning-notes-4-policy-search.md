# Lecture 5 - Policy Search
Difference from lecture 3 and 4: 
- Previously we talked about policy evaluation -> Deep Q Learning based on parameterized function as approximation of $Q(s, a)$. Then the policy is derived from the optimized value, e.g. $\epsilon$-greedy policy.
- In this lecture we will learn how to parameterize the policy itself directly as $\pi_\theta(s, a) = P(a | s, \theta)$.

RL has 
(1) Value Based;
(2) Policy Based;
(3) Actor Critic (combo of Policy based and Value based)

In this lecture we focus on (2).

Key Takeaways:
- **Reinforcement = Reward-Weighted Updates**: The algorithm reinforces actions proportionally to their contribution to the total reward.
- **No Environment Dynamics Required**: Unlike value-based methods (e.g., Q-learning), REINFORCE only needs reward signals and action probabilities, making it model-free.
- **High Variance, Low Bias**: The Monte Carlo approach introduces variance (due to trajectory randomness), but the estimate is unbiased.


## Context of Policy Search vs Policy Evaluation

Reinforcement Learning (RL) algorithms integrate four critical components to enable effective decision-making in dynamic environments:
- **Optimization** drives iterative improvement.
- **Delayed Consequences** ensure long-term planning.
- **Exploration** prevents local optima.
- **Generalization** enables scalability.

### 1. Optimization
- **Role**: Adjust the agent's policy or value functions to maximize cumulative rewards.
- **Mechanism**: Uses algorithms like gradient descent or dynamic programming to update parameters (e.g., Q-values, neural network weights). For example:
  - **Q-learning**: Updates Q-values via the Bellman equation:  
    $$
    Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right].
    $$
  - **Policy Gradient Methods**: Directly optimize the policy using gradient ascent on expected rewards.
- **Purpose**: Ensure continuous improvement toward an optimal strategy.
### 2. Delayed Consequences
- **Role**: Account for actions whose rewards manifest in the future.
- **Mechanism**: Incorporates **temporal credit assignment** using a discount factor $\gamma$ to weigh future rewards:  
  $$
  G_t = \sum_{k=0}^\infty \gamma^k r_{t+k}.
  $$
### 3. Exploration
- **Role**: Discover high-reward actions by testing unfamiliar strategies.
- **Mechanism**: Balances **exploration** (trying new actions) and **exploitation** (using known best actions):
  - **Epsilon-Greedy**: Randomly explore with probability $\epsilon$.
  - **Thompson Sampling**: Probabilistically selects actions based on uncertainty.
  - **Upper Confidence Bound (UCB)**: Favors under-explored actions with high potential.
- **Purpose**: Avoid suboptimal policies by gathering diverse experiences.
### 4. Generalization
- **Role**: Apply learned knowledge to unseen states.
- **Mechanism**: Uses function approximation (e.g., neural networks) to estimate values/policies across large/continuous state spaces.
- **Purpose**: Scale to complex environments (e.g., robotics, game AI) without memorizing every state.

### **Interaction in RL Algorithms**
1. **Agent-Environment Loop**:
   - Observes state $s_t$.
   - Selects action $a_t$ (exploration vs. exploitation).
   - Receives reward $r_t$ and transitions to $s_{t+1}$.
2. **Update Policy/Values**:
   - Optimize using observed rewards and states (handling delayed consequences via $\gamma$).
   - Generalize to new states using function approximators.

## Policy Search
The goal is to maximize the **expected return**:
$$
V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t \Big| s_0 = s, \pi\right],
$$
where trajectories $ t = (s_0, a_0, r_0, s_1, a_1, r_1, \dots) $ are generated under policy $\pi_\theta$.

Symplifying the equation by assuming the trajectory is finite, then we can ignore discounting for now and choose $\gamma=1$. Then the equation becomes
$$
V(s_0, \theta) = \sum_t P(t, \theta)R(t)
$$

The gradient of $ V(\theta) $ w.r.t. $\theta$ becomes:
$$
\nabla_\theta V(\theta) = \nabla_\theta \sum_t P(t; \theta) R(t),
$$
where $ P(t; \theta) $ is the probability of trajectory $ t $, and $ R(t) $ is its cumulative reward. $V^\pi$ is also annotated as $\sum_a \pi(a | s)Q(s, a)$.

## Simplify the Gradient Descent Objective

The critical steps include:
1. Using the log-derivative trick to handle the gradient of a distribution.
2. Decomposing the trajectory probability into policy and environment terms.
3. Approximating the expectation with Monte Carlo sampling.

The final approximation is the foundation of REINFORCE:
$$
\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(t^{(i)}) \nabla_\theta \log P(t^{(i)}),
$$

The term "REINFORCE" in the REINFORCE algorithm is rooted in its mechanism of reinforcing actions that lead to higher rewards by scaling their probability gradients with the observed reward.

The next step is to calculate $\nabla_\theta \log P(t^{(i)})$. a.k.a. score function.

---

### **1. Log-Derivative Trick**
To handle the gradient of a probability distribution, we use the identity:
$$
\nabla_\theta P(t; \theta) = P(t; \theta) \nabla_\theta \log P(t; \theta).
$$
This converts the gradient into an expectation:
$$
\nabla_\theta V(\theta) = \sum_t P(t; \theta) R(t) \nabla_\theta \log P(t; \theta).
$$

---

### **2. Trajectory Probability Decomposition**
The trajectory probability $ P(t; \theta) $ can be decomposed into:
$$
P(t; \theta) = p(s_0) \prod_{k=0}^{T-1} \pi_\theta(a_k | s_k) p(s_{k+1} | s_k, a_k),
$$
where $ p(s_{k+1} | s_k, a_k) $ are environment dynamics. Taking the logarithm:
$$
\log P(t; \theta) = \log p(s_0) + \sum_{k=0}^{T-1} \log \pi_\theta(a_k | s_k) + \log p(s_{k+1} | s_k, a_k).
$$
**Crucially**, the gradient $\nabla_\theta \log P(t; \theta)$ **only depends on the policy**:
$$
\nabla_\theta \log P(t; \theta) = \sum_{k=0}^{T-1} \nabla_\theta \log \pi_\theta(a_k | s_k),
$$
because $ p(s_0) $ and $ p(s_{k+1} | s_k, a_k) $ are independent of $\theta$.

---

### **3. Monte Carlo Approximation**
The expectation $\sum_t P(t; \theta) R(t) \nabla_\theta \log P(t; \theta)$ is approximated by sampling $ m $ trajectories (because of Law of Large Numbers):
$$
\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(t^{(i)}) \nabla_\theta \log P(t^{(i)}).
$$
Substituting the decomposed gradient:
$$
\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(t^{(i)}) \sum_{k=0}^{T-1} \nabla_\theta \log \pi_\theta(a_k^{(i)} | s_k^{(i)}).
$$
This is the **REINFORCE algorithm**, where each action’s log-probability gradient is weighted by the trajectory’s total reward.

## Score Function
The **score function** is a key concept in policy gradient methods. It measures how the log-probability of an action changes with respect to the policy parameters $\theta$. Formally, for a policy $\pi_\theta(a|s)$, the score function is:
$
\nabla_\theta \log \pi_\theta(a|s).
$

This gradient tells us how to adjust $\theta$ to increase/decrease the likelihood of action $a$ in state $s$, which is central to policy optimization.

---

### **2. Typical Score Functions for Common Policies**
#### **a. Softmax Policy (Discrete Actions)**
- **Policy Definition**:  
  The softmax policy converts action preferences ("logits") into probabilities using the softmax function. Let $f(s, a; \theta)$ be a scoring function (e.g., linear or neural network). The probability of action $a$ is:
  $
  \pi_\theta(a|s) = \frac{e^{f(s, a; \theta)}}{\sum_{a'} e^{f(s, a'; \theta)}}.
  $
- **Score Function**:  
  $
  \nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta f(s, a; \theta) - \mathbb{E}_{a' \sim \pi_\theta} \left[ \nabla_\theta f(s, a'; \theta) \right].
  $
  - **Intuition**: The gradient is the difference between the score of the chosen action $a$ and the average score of all actions under $\pi_\theta$.

- **Example**:  
  If $f(s, a; \theta) = \theta^T \phi(s, a)$ (linear preferences), then:
  $
  \nabla_\theta \log \pi_\theta(a|s) = \phi(s, a) - \mathbb{E}_{a' \sim \pi_\theta} \left[ \phi(s, a') \right].
  $
  This is the feature vector of action $a$ minus the expected feature vector under $\pi_\theta$.

#### **b. Gaussian (Normal) Policy (Continuous Actions)**
- **Policy Definition**:  
  For continuous actions (common in robotics), the policy is often a Gaussian distribution with mean $\mu(s; \theta)$ and variance $\sigma^2$ (fixed or learned):
  $
  \pi_\theta(a|s) = \mathcal{N}(a; \mu(s; \theta), \sigma^2).
  $
- **Score Function**:  
  $
  \nabla_\theta \log \pi_\theta(a|s) = \frac{(a - \mu(s; \theta))}{\sigma^2} \nabla_\theta \mu(s; \theta).
  $
  - If $\sigma$ is parameterized, add a term for $\nabla_\theta \sigma$.

- **Intuition**: Adjust $\theta$ to move the mean $\mu(s; \theta)$ closer to actions $a$ that yield higher rewards.

#### **c. Categorical Policy (Discrete Actions)**
- **Policy Definition**:  
  A generalization of the softmax policy, where probabilities are directly parameterized (e.g., a neural network outputting probabilities).
  $
  \pi_\theta(a|s) = p_a, \quad \text{where } \sum_a p_a = 1.
  $
- **Score Function**:  
  For a categorical distribution, the score function is similar to softmax but can be simpler if probabilities are explicitly parameterized:
  $
  \nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}.
  $

---
#### **Practical Implications**
- In **deep learning**, $f(s, a; \theta)$ is often the output of a neural network. The score function is computed via backpropagation.
- The term $\mathbb{E}_{a' \sim \pi_\theta} \left[ \nabla_\theta f(s, a'; \theta) \right]$ introduces **variance reduction** by centering the gradient around the mean.

---

### **3. Why the Score Function Matters**
In policy gradient algorithms (e.g., REINFORCE), the update rule is:
$
\Delta \theta \propto R(t) \nabla_\theta \log \pi_\theta(a|s),
$
where:
- $R(t)$ is the return (total reward) of the trajectory.
- The score function $\nabla_\theta \log \pi_\theta(a|s)$ determines **how to adjust $\theta$** to increase the probability of action $a$.

For example:
- If $R(t)$ is large and positive, actions in the trajectory are "reinforced" (their probabilities increase).
- If $R(t)$ is negative, probabilities decrease.

## Reduce Variance
MC approximation is unbiased but very noisy:
$$
\nabla_\theta V(\theta) \approx \frac{1}{m} \sum_{i=1}^m R(t^{(i)}) \sum_{k=0}^{T-1} \nabla_\theta \log \pi_\theta(a_k^{(i)} | s_k^{(i)}).
$$

Better estimate of the gradient with less data:
- Temporal Structure
- Baseline

### Temporal Structure
We can repeat the same argument to derive the gradient estimator for a single reward term $r_{t'}\$, where $t'$ is a single timestep within a trajectory:
$$
\nabla \mathbb E[r_{t'}] = \mathbb E[r_{t'}\sum_{t=0}^{t'}\nabla _\theta log\pi_\theta(a_t | s_t)]
$$

For each $r_{t'}$, we consider all actions $a_0, ..., a_{t'}$ that influenced $r_{t'}$. This is the essence of temporal structure: understanding how actions at earlier timesteps affect later rewards, and time only flows one way. 

Then we can rewrite the value function
$$
\nabla_\theta V(\theta) \approx \sum_{r'=1}^{T
'} \nabla_{\theta}\mathbb E[r_{t'}]
$$
$$
\nabla_{\theta}\mathbb E[r_{t'}] = \mathbb E[ r_{t'}\sum_{t=0}^{t'} \nabla _\theta log\pi_\theta(a_t | s_t)]
$$

Combining these 2 equations, we get:
$$
\nabla_{\theta}V(\theta) = \mathbb E[ \sum_{t'=1}^{T
'}r_{t'}\sum_{t=0}^{t'} \nabla _\theta log\pi_\theta(a_t | s_t)]
$$

Imagine a matrix with $r_{t'}\nabla _\theta log\pi_ \theta(a_t|s_t)$. where the row index is $t$ and column index is $t'$, if we sum up columns first it is the equation above, if we sum up rows first, it is the equation below.. 

$$
\nabla _\theta \mathbb{E}[R]
= \mathbb{E}[\sum_{t=0}^{T-1}\nabla_ \theta log \pi _\theta(a_t, s_t)\sum_{t'=t}^{T-1}r_{t' }]
$$

Then the parameter updating logic is very similar to MC estimator if $R(t^{(i)})=\sum_{t'=0}^{T'}r_{t'}$, the only difference is that the reward value $G_t$ is no longer coming from the whole episode but the reward since current timestep.

$$
\nabla \theta_t = \alpha \nabla_ \theta log \pi_ \theta(s_t, a_t)G_t
$$

Therefore, MC is a special case where rewards are summed across the entire trajectory.

**Why Temporal Structure Reduces Variance**

- MC Estimator: Uses the total trajectory reward to weight all actions equally, leading to high variance because distant actions are credited/punished for rewards they didn’t directly influence.
- Temporal Structure: Uses per-timestep rewards, weighting only the actions that actually influenced $r_{t'}$. This reduces variance by localizing credit assignment.

**Trajectory vs. Timestep**  
- **Trajectory ($ t^{(i)} $)**: A full sequence $ (s_0, a_0, r_0, \dots, s_T, a_T, r_T) $.  
- **Timestep ($ t $)**: A single step within a trajectory.  

#### Example of MC policy search vs temporal structure  
Consider a trajectory with $ T=3 $ timesteps:  
$
t^{(i)} = (s_0, a_0, r_0), (s_1, a_1, r_1), (s_2, a_2, r_2).
$
- **MC Gradient**:  
  $$
  \nabla_\theta V(\theta) \propto (r_0 + r_1 + r_2) \left( \nabla \log \pi(a_0|s_0) + \nabla \log \pi(a_1|s_1) + \nabla \log \pi(a_2|s_2) \right)
  $$
- **Temporal Structure Gradient**:  
  $$
  \nabla_\theta V(\theta) \propto r_0 \nabla \log \pi(a_0|s_0) + r_1 \left( \nabla \log \pi(a_0|s_0) + \nabla \log \pi(a_1|s_1) \right) + r_2 \left( \nabla \log \pi(a_0|s_0) + \nabla \log \pi(a_1|s_1) + \nabla \log \pi(a_2|s_2) \right)
  $$
  This attributes each $ r_{t'} $ only to the relevant actions, reducing irrelevant correlations.

One problem is that $G_t$ has high variance. In order to converge as quickly as possible to a local optima, i.e. low variance for the gradient.

### Baseline
Reduce variance by introducing a baseline $b(s)$ (not a function of $\theta$ or $a$, only a function of the state for each given timestep).
$$
 \nabla _\theta \mathbb{E}_T[R] = \mathbb{E}_T[\sum_{t=0}^{T-1}\nabla _\theta log\pi(a_t|s_t; \theta)(\sum_{k=t}^{T-1}r_{k}-b(s_t))]
$$

The main thing to consider is how much better this policy is, coming to other policies we could do.

Near optimal choice is the expected return. i.e. a general estimate of the performance of state $s_t$:
$$
b(s_t) ≈ \mathbb{E}[r_t + r_{t+1} + ... + r_{T-1}]
$$

#### Different Options for Baseline
We can pick the $Q$ value for $G_t$, i.e. $Q^\pi(s, a) = \sum_0^{\infin} \gamma^{i} r_i$.

State-value function $\mathbb{V}(s) = \mathbb{E}[Q^\pi(s, a)]$ can serve as a great baseline for $b(s)$.
