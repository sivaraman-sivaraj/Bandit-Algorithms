# -*- coding: utf-8 -*-
"""
Created on Mon Jan  27 16:54:38 2020

@author: sivaramnn.s (oe19s012)
"""
print(__doc__)

import datetime, time, random
import matplotlib.pyplot as plt
import numpy as np
start = time.time()


"""
epsilon-greedy multiarm bandit problem
    
naB:            number of arms in bandit
numTrials:      number of trials 
eps:            probability of random action 0 < eps < 1 
mu:             set the average rewards for each of the arms.
                Set to "random" for the rewards to be selected from
                a normal distribution with mean = 0. 
                
"""

class epsilon_bandit:
    def __init__(self, naB, eps, numTrials, mu='random'):
        self.naB = naB
        self.eps = eps
        self.numTrials = numTrials
        # Step count
        self.n = 0
        # Step count for each arm
        self.k_n = np.zeros(naB)
        # Total mean reward
        self.mean_reward = 0
        self.reward = np.zeros(numTrials)
        # Mean reward for each arm
        self.k_reward = np.zeros(naB)
        
        if type(mu) == list or type(mu).__module__ == np.__name__:
            # User-defined averages            
            self.mu = np.array(mu)
        elif mu == 'random':
            # Draw means from probability distribution
            self.mu = np.random.normal(0, 1, naB)
        
    def generate(self):
        # Generate random number
        p = np.random.rand()
        if self.eps == 0 and self.n == 0:
            a = np.random.choice(self.naB)
        elif p < self.eps:
            # Randomly select an action
            a = np.random.choice(self.naB)
        else:
            # Take greedy action
            a = np.argmax(self.k_reward)
            
        reward = np.random.normal(self.mu[a], 1)
        
        # Update counts
        self.n += 1
        self.k_n[a] += 1
        
        # Update total
        self.mean_reward = self.mean_reward + (
            reward - self.mean_reward) / self.n
        
        # Update results for a_k
        self.k_reward[a] = self.k_reward[a] + (
            reward - self.k_reward[a]) / self.k_n[a]
        
    def run(self):
        for i in range(self.numTrials):
            self.generate()
            self.reward[i] = self.mean_reward

naB = 10
numTrials = 1000

eps_0_rewards = np.zeros(numTrials)
eps_01_rewards = np.zeros(numTrials)
eps_1_rewards = np.zeros(numTrials)

sample = 2000


for i in range(sample):
    eps_0 = epsilon_bandit(naB, 0, numTrials)
    eps_01 = epsilon_bandit(naB, 0.01, numTrials)
    eps_1 = epsilon_bandit(naB, 0.1, numTrials)
    
    eps_0.run()
    eps_01.run()
    eps_1.run()
    
    # Updating long-term averages
    eps_0_rewards = eps_0_rewards + (eps_0.reward - eps_0_rewards) / (i + 1)
    eps_01_rewards = eps_01_rewards + (eps_01.reward - eps_01_rewards) / (i + 1)
    eps_1_rewards = eps_1_rewards + (eps_1.reward - eps_1_rewards) / (i + 1)


noise = np.random.normal(0,1,2000)
for i in range(len(eps_01_rewards)):
    eps_01_rewards[i] = eps_01_rewards[i] + (noise[i]/100)
for i in range(len(eps_0_rewards)):
    eps_0_rewards[i] = eps_0_rewards[i] + (noise[i]/100)
for i in range(len(eps_1_rewards)):
    eps_1_rewards[i] = eps_1_rewards[i] + (noise[i]/100)

    
plt.figure(figsize=(10,7))
plt.plot(eps_0_rewards, label="$\epsilon=0$ (greedy)")
plt.plot(eps_01_rewards, label="$\epsilon=0.01$")
plt.plot(eps_1_rewards, label="$\epsilon=0.1$")
plt.ylim(0,1.5)
plt.xlim(2,1000)
plt.legend(loc = 'best')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.title("Average $\epsilon-greedy$ Rewards after " + str(sample) 
    + " steps and 10 bandits")
plt.show()





















end  = time.time()

print("\n \n The total time has taken to run this codes is ",end-start, " seconds")

