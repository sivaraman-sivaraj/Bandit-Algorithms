# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 18:41:53 2020

@author: sivaraman sivaraj
"""

import math, time, random
import matplotlib.pyplot as plt
import numpy as np
start = time.time()

# Implementing UCB
"""
Upper Confindence Bound algorithm
    
naB:            number of arms in bandit
numTrials:      number of trials 

"""
bandits  = 2000
numTrials = 1000
naB = 10
true_values = np.random.randn(bandits, naB)
print(true_values)
def UCB_generation(bandits,numTrials,naB,true_values):
    picks = []
    for i in range(bandits):
        pick = []
        estimate = np.random.normal(true_values[i], 1)
        count = [1]*naB
        bands = np.array(estimate[:]) + (2*np.log(naB)/count[1])**0.5
        pick = pick + list(bands)
        
        for k in range(naB, numTrials+naB):
            action = np.argmax(bands)
            Reward = np.random.normal(true_values[i][action], 1)
            estimate[action] = (estimate[action]*count[action] + Reward)/(count[action]+1)
            
            for l in range(naB):#updating the value
                bands[l] = estimate[l] + (2*np.log(k)/count[l])**0.5
            pick.append(Reward)
        picks.append(pick)
    avg = np.mean(picks, axis = 0)[naB:]
    return avg

def plotGraph(avg):
    plt.figure(figsize = (10,7))
    plt.plot(list(range(1000)), avg)
    plt.xlabel('number of trials')
    plt.ylabel('Average Reward')
    plt.axhline(y=1.0)
    plt.ylim(1,1.7)
    plt.title('UCB1 for 1000 arm bandits')
    plt.legend()
    plt.show()
          
v = UCB_generation(bandits,numTrials,naB,true_values)
plotGraph(v)  
            

end  = time.time()

print("\n \n The total time has taken to run this codes is ",end-start, " seconds")
            
            
            
            