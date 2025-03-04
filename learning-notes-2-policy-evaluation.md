# Lecture 3 - Model-free Policy Evaluation
Model-free Policy: Policy Evaluation with no knowledge of how the world works ($P(s'|s)$ and $r(s)$):
- Monte Carlo Learning
- Temporal Difference (TD) Learning

The model-free policy takes the action from $argmax_a$ for each state during each iteration. It means that $\pi_{t+1}$ is deterministic and we cannot get $Q^{\pi+1}(s, a)$ for actions who is not picked.

## Policy Evaluation
Computational Complexity / Cost

Memory Requirement

Statistical Efficiency

Empirical Accuracy
- $Bias_\theta(\hat\theta) = E_{x|\theta}(\hat\theta) - \hat\theta$
- $Var(\hat\theta) = E_{x|\theta}[(\hat\theta - E(\hat\theta))^2]$
- MSR is a common measurement: $Var(\hat\theta) + Bias_\theta(\hat\theta)$

Consistency: whether policy can converge
The estimator $\hat\theta_n$ is consistent if for all $\epsilon > 0$

$$
lim_{n -> \infin} P_r(|\hat\theta_n - \theta| > \epsilon) = 0
$$

## Monte Carlo Policy Evaluation
Monte Carlo methods are model-free algorithms that learn from complete episodes of experience. They don't require knowledge of the environment's dynamics and instead rely on averaging sample returns. So, they are useful for episodic tasks where the agent can experience many episodes. Types:

### First-visit MC

- Only the first occurrence of a state in an episode is used.

- Unbiased estimator.

- Lower computational cost per episode compared to every-visit.

### Every-visit MC

- Every occurrence of a state in an episode is used.

- Biased estimator because of correlated visits.

- Higher computational cost but might converge faster in some cases.

### Incremental MC

- A method to update averages without storing all returns.

- Uses formulas like $V(s_t) = V(s_t) + (G_t - V(s_t))/N(S_t)$, where $G$ is the observed return. Should always use for efficiency, regardless of first-visit or every-visit.

# Temporal Difference Policy Evaluation

TD learning is a combination of **Monte Carlo policy** (sampling) and **dynamic programming** (bootstrap) using bellman backup equation. It is model-free, i.e., don't need parameterize reward / dynamic model. 

$$
V^\pi(s_t) = V^\pi(s_t) + \alpha([r_t + \gamma V^\pi(s_{t+1})] - V^\pi(s_t))
$$
- Goal: Calculate expected reward for a policy.
- Pros: Immediately update estimate of $V^\pi$ after each $(s, a, r, s')$ tuple. Variance is smaller compared to MC policy.
- Cons: Biased compared to MC policy (because MC policy is the actual reward after the episode reaches terminate state).

$[r_t + \gamma V^\pi(s_{t+1})]$ is the target (estimate based on next state instead of the whole episode). The difference between this target and the current value estimate is `TD(0)` error:
$\epsilon_t = r_t + \gamma V^\pi(s+1)-V^\pi(s_t)$.

pros: after $s_{t+1}$ is observed, $V^\pi(s_t)$ can be updated immediately.

TD target $r_t + \gamma V(s_{t+1})$ is a biased estimate of $G_t$ since $V(s_{t+1})$ may be inaccurate. But it has lower variance than MC because 1 step has lower variance than multiple step in a single episode.

Converge condition:
- $\sum^\infin_{n=1}\alpha_n(s_j) = \infin$ (allow the algorithm exploring the whole parameter space).
- $\sum^\infin_{n=1}\alpha_n^2(s_i) < \infin$ (ensure noise is diminishing over time, allowing stabilization variance. Noise refers to the variability in updates caused by using random samples).

# Certainty Equivalence
Certainty Equivalence assumes that the estimated model (transition probabilities and rewards) of the environment is accurate. The optimal policy is derived under this assumption, ignoring uncertainty in the model.

It is a model based option. For policy eval without the models, recompute max likelihood MDP model for (s, a):
- Estimate $\hat P(s' | s, a)$ and $r(s, a)$ from samples.


# Batch Policy
Batch Policy Learning uses a fixed dataset (batch) of experiences to learn a policy.

Key Idea: No new interactions with the environment; learning occurs offline.

Connection to Certainty Equivalence: Batch methods often use the dataset to compute model parameters (e.g., via maximum likelihood) and then apply certainty equivalence to derive the policy.

Context of Likelihood: 
- The likelihood function L(θ | data) is proportional to P(data | θ). The key difference is that in probability, the parameters are fixed, and we consider the distribution over possible data, while in likelihood, the data are fixed, and we consider the function over possible parameters.
- Likelihood on its own doesn't assign probabilities to parameters. It's a measure of relative plausibility.
