### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from assignment1.lake_envs import *
import matplotlib.pyplot as plt
from tqdm import *

def addNoise(q,s0,env):
    ind = (q[s0] == np.max(q[s0]))
    addnoise = np.arange(env.nA)
    np.random.shuffle(addnoise)
    cur = q.copy()
    cur[s0][ind] = q[s0][ind] + addnoise[ind]
    return cur[s0]

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state, action values
    """
    q = np.zeros((env.nS, env.nA))
    for _ in range(num_episodes):
        done = False
        s0 = env.reset()
        while not done:
            if np.random.uniform(0,1,1)[0] < e:
                #greedy
                action = np.random.choice(list(env.P[s0].keys()), 1)[0]
            else:
                action = np.argmax(addNoise(q,s0,env))
            s1, r, done, info = env.step(action)
            q[s0][action] = q[s0][action] + lr*(r + gamma*np.max(q[s1]) - q[s0][action])
            s0 = s1

        if _ % 10 == 0:
            e *= decay_rate
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    return q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
    """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
    Update Q at the end of every episode.

    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    decay_rate: float
        Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    np.array
        An array of shape [env.nS x env.nA] representing state-action values
    """
    q = np.zeros((env.nS, env.nA))
    for _ in range(num_episodes):
        done = False
        s0 = env.reset()
        if np.random.uniform(0, 1, 1)[0] < e:
            a0 = np.random.randint(0, env.nA, 1)[0]
        else:
            #a0 = np.argmax(q[s0])
            a0 = np.argmax(addNoise(q, s0, env))

        while not done:
            s1, r, done, info = env.step(a0)
            if np.random.uniform(0, 1, 1)[0] < e:
                a1 = np.random.randint(0, env.nA, 1)[0]
            else:
                #a1 = np.argmax(q[s1])
                a1 = np.argmax(addNoise(q, s1, env))
            q[s0][a0] = q[s0][a0] + lr * (r + gamma * q[s1][a1] - q[s0][a0])
            s0 = s1
            a0 = a1
        if _ % 10 == 0:
            e *= decay_rate
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################

    return q

def render_single_Q(env, Q):
    """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play Q function on. Must have nS, nA, and P as
        attributes.
    Q: np.array of shape [env.nS x env.nA]
        state-action values.
    """

    episode_reward = 0
    state = env.reset()
    done = False
    #while not done:
    for t in range(100):
        #env.render()  #show frames
        #time.sleep(0.5) # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        episode_reward += reward
        if(done):break
    #print "Episode reward: %f" % episode_reward
    return episode_reward
    
# Feel free to run your own debug code in main!
def main():
    env = gym.make('Stochastic-4x4-FrozenLake-v0')
    #env = gym.make('Deterministic-4x4-FrozenLake-v0')

    score1 = []
    score2 = []
    average_score1 = []
    average_score2 = []
    all_epoch = 200
    for i in tqdm(range(all_epoch)):
        Q1 = learn_Q_QLearning(env, num_episodes=i+1)
        Q2 = learn_Q_SARSA(env, num_episodes=i+1)
        episode_reward1 = render_single_Q(env, Q1)
        episode_reward2 = render_single_Q(env, Q2)
        score1.append(episode_reward1)
        score2.append(episode_reward2)
    for i in range(all_epoch):
        average_score1.append(np.mean(score1[:i+1]))
        average_score2.append(np.mean(score2[:i+1]))
    plt.plot(np.arange(all_epoch),np.array(average_score1))
    plt.plot(np.arange(all_epoch),np.array(average_score2))
    plt.title('The running average score of the Q-learning agent')
    plt.xlabel('traning episodes')
    plt.ylabel('score')
    plt.legend(['q-learning', 'sarsa'], loc='upper right')
    plt.show()
    plt.savefig('model-free.jpg')
           
if __name__ == '__main__':
    main()
