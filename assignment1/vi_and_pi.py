### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import gym.spaces
import time
from assignment1.lake_envs import *
import random

np.set_printoptions(precision=3)


def isover(V, V_new, tol):
    if np.all(np.abs(V - V_new) < tol):  # np.sum(np.sqrt(np.square(V_new-V))) < tol
        return 1
    return 0


def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    policy: np.array
        The policy to evaluate. Maps states to actions.
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns
    -------
    value function: np.ndarray
        The value function from the given policy.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    #for i in range(nS):
    #    for j in range(nA):
    #        print(i,j,P[i][j])
    v = np.zeros(nS)
    for i in range(max_iteration):
        v_new = np.zeros(nS, dtype=float)
        for si in range(nS):
            for pro, sj, r, t in P[si][policy[si]]:
                v_new[si] += pro * (r + gamma * v[sj])
                #print(si, v_new[si],r)
        #print('v new',v_new)
        #print('v',v)
        if (np.abs(v_new - v) < tol).all(): break
        v = v_new.copy()

    return v


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new policy: np.ndarray
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    q = np.zeros((nS, nA))
    for ai in range(nA):
        for si in range(nS):
            for pro, sj, r, t in P[si][ai]:
                q[si][ai] += pro * (r + gamma * value_from_policy[sj])

    ind = (q == np.max(q,axis=1,keepdims=True))
    addnoise = np.arange(nS*nA).reshape(q.shape)
    np.random.shuffle(addnoise)
    q[ind] += addnoise[ind]
    policy = np.argmax(q, axis=1)
    # if there is no noise, cannot find optimal result when there is Stochastic-4x4-FrozenLake-v0
    # even add noise, sometimes, the player will go into the hole
    #print(policy)
    #print('add noise edd')
    return policy


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

    You should use the policy_evaluation and policy_improvement methods to
    implement this method.

    Parameters
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    pi0 = np.random.randint(0, nA, (nS))
    pi0 = np.ones(nS, dtype=int)*2
    pi1 = pi0.copy()
    #print(pi0)
    i = 0
    while i < max_iteration and (i < 10 or (np.abs(pi0 - pi1) > tol).any()):
        pi0 = pi1.copy()
        v = policy_evaluation(P, nS, nA, pi1, gamma, max_iteration, tol)
        i += 1
        pi1 = policy_improvement(P, nS, nA, v, pi1, gamma)
        #print(pi0)
    return v, pi1


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P: dictionary
        It is from gym.core.Environment
        P[state][action] is tuples with (probability, nextstate, reward, terminal)
    nS: int
        number of states
    nA: int
        number of actions
    gamma: float
        Discount factor. Number in range [0, 1)
    max_iteration: int
        The maximum number of iterations to run before stopping. Feel free to change it.
    tol: float
        Determines when value function has converged.
    Returns:
    ----------
    value function: np.ndarray
    policy: np.ndarray
    """
    v = np.zeros(nS, dtype=float)
    for i in range(max_iteration):
        q = np.zeros((nS, nA), dtype=float)
        vpre = v.copy()
        for s in range(nS):
            for a in range(nA):
                for pro, sj, r, terminal in P[s][a]:
                    q[s][a] += pro * (r + gamma * v[sj])
        v = np.max(q,axis=1)
        if(np.abs(vpre - v)<tol).all():break
    policy = policy_improvement(P, nS, nA, v, np.zeros(nS, dtype=int), gamma)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    return v, policy


def example(env):
    """Show an example of gym
    Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
    """
    env.seed(0);
    from gym.spaces import prng;
    prng.seed(10)  # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render();


def render_single(env, policy):
    """Renders policy once on environment. Watch your agent play!

        Parameters
        ----------
        env: gym.core.Environment
            Environment to play on. Must have nS, nA, and P as
            attributes.
        Policy: np.array of shape [env.nS]
            The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(100):
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    assert done
    env.render();
    print("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    #env = gym.make('Deterministic-4x4-FrozenLake-v0')
    # print env.__doc__
    # print "Here is an example of state, action, reward, and next state"
    # example(env)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=600, tol=1e-3)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=600, tol=1e-3)
    print('Policy Iteration')
    print('  Optimal Value Function: %r' % V_pi)
    print('  Optimal Policy:         %r' % p_pi)
    print('Value Iteration')
    print('  Optimal Value Function: %r' % V_vi)
    print('  Optimal Policy:         %r' % p_vi)
    #render_single(env, p_pi)
    render_single(env,p_vi)
    print('\n##########\n##########\n\n')
