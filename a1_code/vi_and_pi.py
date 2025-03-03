### MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim
import copy 

np.set_printoptions(precision=3)

def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.

    Parameters
    ----------
    state: int
    action: int
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    V: np.array (num_states)

    Returns
    -------
    backup_val: float
    """
    backup_val = 0.
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    future_reward = np.dot(V, T[state][action])
    # V(s') = sum[Prob_pi(a|s') * Q_pi(s', a)]
    # Return Q_pi(s, a) = R(s, a) + gamma * sum_s'(P[s'|s, a] * V(s'))
    backup_val = R[state][action] + gamma * future_reward

    return backup_val

def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    gamma: float
    tol: float

    Returns
    -------
    value_function: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        new_value = np.zeros(num_states)
        for s in range(num_states):
            # stochastic policy, V_pi(s) = sum_a[prob(a|s) * Q(s, a)]
            # deterministic policy, V_pi(s) = Q(s, pi(a))
            q_s_a = bellman_backup(s, policy[s], R, T, gamma, value_function)
            new_value[s] = q_s_a
        
        if max(abs(value_function - new_value)) < tol:
            break
        value_function = new_value.copy()

    ############################
    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given the value function induced by a given policy, perform policy improvement
    Parameters
    ----------
    policy: np.array (num_states)
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)
    V_policy: np.array (num_states)
    gamma: float

    Returns
    -------
    new_policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(num_states):
        Q_s = np.zeros(num_actions)
        for a in range(num_actions):
            Q_s[a] = bellman_backup(s, a, R, T, gamma, V_policy)
        new_policy[s] = np.argmax(Q_s)

    ############################
    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    V_policy: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    V_policy = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #

    while True:
        V_policy = policy_evaluation(policy, R, T, gamma, tol)
        old_policy = policy.copy()
        policy = policy_improvement(old_policy, R, T, V_policy, gamma)
        if np.array_equal(old_policy, policy):
            break
    ############################
    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """Runs value iteration.
    Parameters
    ----------
    R: np.array (num_states, num_actions)
    T: np.array (num_states, num_actions, num_states)

    Returns
    -------
    value_function: np.array (num_states)
    policy: np.array (num_states)
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        new_value = np.zeros(num_states)
        for state in range(num_states):
            bellman_backups = np.zeros(num_actions)
            for action in range(num_actions):
                bellman_backups[action] = bellman_backup(state, action, R, T, gamma, value_function)
            policy[state] = np.argmax(bellman_backups)
            new_value[state] = max(bellman_backups)
        delta = max(abs(value_function - new_value))
        value_function = new_value.copy()
        if delta < tol:
            break
    ############################
    return value_function, policy


# Edit below to run policy and value iteration on different configurations
# You may change the parameters in the functions below
if __name__ == "__main__":
    SEED = 1234

    RIVER_CURRENT = 'STRONG' # 'WEAK' # 'MEDIUM'
    assert RIVER_CURRENT in ['WEAK', 'MEDIUM', 'STRONG']
    env = RiverSwim(RIVER_CURRENT, SEED)

    R, T = env.get_model()
    discount_factor = 0.94
    
    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([['L', 'R'][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([['L', 'R'][a] for a in policy_vi])
