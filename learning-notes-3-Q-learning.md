# Lecture 4 - Deep Q-Learning
Difference from lecture 3: we will try stochastic policy so that different actions are allowed under same state.

## Tradeoff between exploration and exploitation
Determinisitic Policy vs Stochastic Policy
- When using deterministic policy $\pi(a|s)$, we can only get a $Q(s, a)$ of one action when exploring the world. So using stochasitic is better to explore multiple actions at one time.

Policy Evaluation vs Policy Iteration
- $Q(s, a)$ is the key of policy evaluation. But how to apply the learned value estimate to the new action is a tradeoff between exploration and exploitation. Relying on the learned knowledge to take action favors exploitation, trying random actions favors exploration - good for learning but may give poor reward.

## $\epsilon$-greedy Policies
with $A$ as the number of actions.

$\pi (a|s)= arg max_aQ(s, a)$ with prob $1 - \epsilon + \epsilon/|A|$.

$\pi (a|s) \not= arg max_aQ(s, a)$ with prob $\epsilon/|A|$.

i.e. $arg max_aQ(s, a)$ with probability $1 - \epsilon$, else, select action uniformly at random.

It is no longer determinisitc policy given that different actions are allowed.

## Model-free Control - Tabular
Recall Monte Carlo Policy Evaluation

loop k:
- Sample k-th episode ($s_{k,1}, a_{k, 1}, r_{k, 1}, s_{k, 2}, a_{k, 2}, r_{k, 2}, ..., r_{k, t}$) given $\pi$.
- $G(k, t) = r_{k, t} + \gamma r_{k, t+1} + \gamma^2 r_{k, t+2} + ... + \gamma ^{T_i - 1} r_{k, T_i}$ for any $t$ in $T$
- for t = 1, 2, ..., T do
- if First visit to (s, a) in episode k then
- $N(s, a) = N(s, a) + 1$
- $Q(s_t, a_t) = Q(s_t, a_t) + 1/N(s, a)(G_{k, t} - Q(s_t, a_t))$

Monte Carlo control: after the valuation of each episode:
- when updating k to k+1, set $\epsilon = 1/k$, update $\pi_{k} = \epsilon greedy(Q)$

Is $Q$ an estimate of $Q^\pi$? No. Because every time the policy is evaluated, it gets updated and tried the next time only once. $Q$ is a weighted average across many policies, not only $\pi$.

When will this procedure fail to estimate $Q^*$? We will converge to something deterministic because $\epsilon$ will become smaller as $k$ increases. But the thing we will converge to is not clear to be $Q*$.

### GLIE - Proof of converge to $Q^*$
Greedy in the Limit of Infinity Exploration. Property:
- inifite pairs of ($s, a$): $lim_{i->\infin}N(s, a) -> \infin$
- Behavior policy (policy used to act in the world) converges to greedy policy: $lim_{i->\infin}\pi(a|s)-> arg max_a Q(s, a)$ with probability 1.

### Temporal Difference (TD) Evaluation - Policy Update
**On-policy learning**: Direct experience. Learn to estimate and evaluate a policy from experience obtained from following that policy 
**Off-policy learning**: Learn to estimate and evaluate a policy using experience obtained from following a different policy.

### On Policy - SARSA
State - Action - Reward - Next State - Next Action

Every time the next $(a, s)$ pair is observed, the Q value would be updated based on $Q(s_t, a_t) + \alpha (r_t+\gamma Q(s_{t+1}, a_{t+1})) - Q(s_t, a_t)$. Then perform policy improvement based on $\epsilon$-greedy policy.

Convergence Properties of SARSA: 
1) policy sequence $\pi_t(a|s)$ satisfies the condition of GLIE.
2) The step-sizes $\alpha_t$ satisfy the Robbins-Munro sequence such that
$$
\sum^\infin_{t=1}\alpha_t = \infin
$$
$$
\sum^\infin_{t=1}\alpha_t^2 < \infin
$$
### Off Policy - Q-Learning
Directly estimate the value of $\pi^*$ instead of the behavior policy:
- $Q(s_t, a_t) + \alpha (r_t+\gamma max_a'Q(s_t, a')) - Q(s_t, a_t)$

The calculation is based on the best reward I could have achieved, regardless of what's the actual action and reward I get in the next state. It results in one important difference: Q-learning **doesn't require GLIE properties**, since the actual action is separated from the estimate and policy update direction.

Side Notes:  "ties" refer to situations where multiple actions in a given state have the same highest Q-value.

## Function Approximation

Motivation of Function Approximation:
- When you can write values for action and state separately, tabular is a good fit. But when action and state space is continuous or too large, we want more compact representation that generalizes across states and actions to avoid explicitly storing or learning the results $(P, V, Q, \pi)$ for every state or state and action.

Pros:
- Reduce memory needed to store $(P, R) / V / Q / \pi$.
- Reduce computation needed to compute $(P, R) / V / Q / \pi$.
- Reduce experience needed to find a good $(P, R) / V / Q / \pi$.

e.g. in video game, you don't want to learn adjacent pixels from scratch, through compact representation, you might still take the same decision through generalization.

Function Approximation Definition: Assume we have an oracle that return the true value for $Q^\pi(s, a)$, we can approximate the oracle using parameterized functions to provide the best estimation of $Q^\pi(s, a)$.

How to Optimize the Approximate - Stochastic Gradient Descent using MSE.

### Model Free Policy Evalue + Function Approximate
We visited model free policy evaluation with tabular Q values. Now let's revisit these policy evaluation method but with function approximate.

Main Difference:
- Besides evaluate policy $\pi(s)$ and update the policy to use either random $a$ or $a$ to maximize Q value.
- We also update the approximate function of $Q(s, a)$.

### Stability Issue
**Deadly Triad** can lead to oscillations or lack of convergence (because value function approximation fitting can be an expansion, which means the distance between the approximation and the oracle might become larger and larger):
- bootstrapping
- function approximation
- off policy learning (e.g. Q-learning)

Two paticular issues:
1. Correlation between samples, not i.i.d.
2. Non-stationary TD learning $r + \gamma V(s')$. $V(s')$ is constantly changing as the more samples.

Using experience replay and fix Q target - both things make big difference especially one of them.


Stabilization Techniques

a. Target Network
Prevents moving targets by freezing 
$θ$ during updates. Imagine that a supervised learning $y = \hat f(x)$, if $y$ is constantly changing, the model is more unstable. Changing the $Q$ function changes $y$ because it uses as part of the target.

b. Experience Replay (**very important for DQN**)
Breaks temporal correlation by sampling random transitions. Balances exploration and exploitation.

c. Gradient Clipping
Limits gradient magnitudes to avoid parameter divergence.

### Monte Carlo Evaluation + Function Approximate
$G(s, a)$ serves as the oracle of $Q$, i.e. the observed return based on $r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}...$.

Then the gradient comes from derivitive of MSE to calculate new $Q$ becomes:
$$
\nabla J(s_t, a_t) = - 2(G(s_t, a_t) - \hat Q(s_t, a_t; w)) \nabla\hat Q_w(s_t, a_t; w)
$$
$$
\delta w = 1/2\alpha*\nabla_w J(s_t, a_t)
$$

In this way, we can evaluate a fixed MC policy given k episode to get a function approximate of $Q$. Then we can use it to do policy improvement/control.
### TD Learning + Function Approximate
Objective: Minimize Temporal Difference (TD) Error, i.e.
$$
L(\theta) = E[(r + \gamma max_{a'}Q(s', a'; \theta^-) - \hat Q(s, a; \theta))^2]
$$
Key components:
- Online network $\hat Q(s, a; \theta)$: Actively updated via SGD. Predicts  Q-values for the current state-action pair.
- Target network $\hat Q(s', a'; \theta^-)$: A frozen copy of the online network, updated periodically. It computes the target Q-value $r + \gamma max_{a'}\hat Q(s', a'; \theta^-)$

$\theta^-$: Parameters of a target network (fixed during updates for stability).
