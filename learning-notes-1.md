# Key Takeaways from Lecture 1 & 2
## Markov Assumption
Assume the next state is only correlated with the current state and current action, i.e.

$$
P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid h_t, a_t)
$$

## MDP (Markov decision process) Model
MDP is a dynamic model consisting of 4 tuple $(State, Action, Reward, Probability)$ to model the interaction between a learning agent and its environment. It is derived from Markov assumption. 

- $P(s_{t+1} = s' | s_t = s, a_t = a)$

- Initial $V_O(s) = 0$ for all s because we don't know $V^*$.

- For `k = 1` until convergence: For all `s` belongs to `S`, the value of a state is the immediate reward plus a discounted accumulated reward in the future: $V_k(s) = R(s) + \gamma \sum_{s' \in S} V_{k-1}(S')$


MDP + policy = Markov Reward Process (MRP): $(S, R^\pi, P^\pi, \gamma)$. Policy specifies action to take in each state: $\pi(a|s)$.
- Reward of a state: $R^\pi = \sum_{a \in A}\pi(a|s) * R(s, a)$
- Dynamic model of state transition: $P^\pi(s'|s) = \sum_{a \in A}\pi(a|s)P(s'|s, a)$
- $\pi^* = argmax_{\pi}V^\pi(s)$
- State action value of a policy: $Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in S}(P(s'|s)  V^\pi(s'))$. It means the expected reward if we take action `a` then follow policy $\pi$ for follow up actions for future state $s'$. $V^\pi(s)$ is the value of state `s'` following policy $\pi$, which equals to $\sum_{a \in A} Q^\pi(s, a)$.

The policy improvement follows the action that gives the maximum reward, i.e. $R(s_{i}, \pi_{i+1}(s)) = max_aR(s, a)$.

## Bellman Equations
- $$V^\pi(s) = \sum_{a \in A}\pi(a|s) [R(s, a) + \gamma\sum_{s' \in S}P(s'|a)V^\pi(s')]$$
- $$Q^\pi(s, a) =  R(s, a) + \gamma\sum_{s' \in S}P(s'|a)\sum_{a' in A}\pi(a'|s)Q^\pi(s', a)$$

Q values are more directly useful for decision making since it values each action given a state.

## Bellman Backup
Goal: compute the optiomal value function $V^\pi(s)$ or $Q^\pi(s, a)$.
Definition: Update rule that refines the estimated value of a state by incorparating the immediate reward and expected discounted future reward.

"Backup" means applying the expected reward from future states based on the current policy back to the current policy to derive the next policy.

Value iteration:
- Iterate on the value until it converges. i.e. $v'^\pi(s) = max_a[R(s, a) + \sum_{s' \in S}P(s'|s, a)V^\pi(s')]$.
- Then get the policy of actions for each state who give maximum value.

Policy iteration: 
- compute $Q^\pi(s, a) = R(s, a) + \gamma\sum_{s' \in S}P(s' | s, a) V^\pi(s')$
- $\pi'(s) = arg max_aQ^\pi(s, a)$

These 2 types of iterations give the same results. It's just that the value iteration takes a shortcut to converge on an optimal value along with the optimal policy.
