# -*- coding: utf-8 -*-
"""
Created on Mon Jan  27 18:28:07 2020

@author: sivaramnn.s (oe19s012)
"""

print(__doc__)

import datetime, time, random
import matplotlib.pyplot as plt
import numpy as np
start = time.time()


"""
softmax algorithm
    
naB:            number of arms in bandit
numTrials:      number of trials 

"""
bandits = 2000
naB = 10
numTrials = 1000
T = [0.02, 0.04, 0.4, 1] # Temperature values
color = ['b', 'g', 'r', 'k']

true_values = np.random.randn(bandits, naB)

average = []


def softmax_generation(T,bandits,numTrials,average,color,naB):
    for t in T:

        picks = []
        for i in range(bandits): # Running the Softmax algorithm 
            estimate = np.random.normal(true_values[i], 1) # Initialising the values
            count = [1]*10 # Picking up each arm once

            pick = [] 
            for k in range(numTrials):

                Probab = np.array(list(map(np.exp, np.array(estimate)/t))) #  softmax function elements
                Probab = Probab/np.sum(Probab) # softmax function
    
                action = int(np.random.choice(list(range(naB)), 1, p=Probab))
                Reward = np.random.normal(true_values[i][action], 1) # Finding the reward

                estimate[action] = (estimate[action]*count[action] + Reward)/(count[action]+1) # Updating 
                count[action]+=1
                pick.append(Reward)
      
            picks.append(pick) 

        picks = np.array(picks) 
        average.append(np.mean(picks,axis = 0)) 
        
def plotgraph():
     plt.figure(figsize=(10,7))
     for temp in range(len(T)):
        plt.plot(list(range(1000)), average[temp], c = color[temp], label = f'T = {T[temp]}') 
        plt.title('Softmax Learning Algorithm')
        plt.xlabel('Time-step')
        plt.ylabel('Average Reward')
        plt.axhline(y=1)
        plt.ylim(0.3,1.8)
        plt.legend(loc = 'best')
     plt.show()


softmax_generation(T,bandits,numTrials,average,color,naB)
plotgraph()



end  = time.time()

print("\n \n The total time has taken to run this codes is ",end-start, " seconds")



