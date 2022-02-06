# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:27:13 2020

@author: sivaraman s (oe19s012)

"""

import numpy as np
import matplotlib.pyplot as plt
import random,math, time
start = time.time()

"""
Meadian elimination algorithm
    
naB:            number of arms in bandit
numTrials:      number of trials 

"""

bandits = 2000
naB = 10

true_values = np.random.normal(size=[bandits,naB])
eps_delta_pairs=[[1.3,0.6,'b'],[0.7,0.4,'r'],[1.2,0.7,'g']]

def MedianElimination(naB,bandits,true_values,eps_delta_pairs):
    plt.figure(figsize = (10,7))
    for pairs in eps_delta_pairs:
        start=time.clock()
        totl_time=0.0
        times=0.0
        eps_frst = pairs[0]
        delta_frst=pairs[1]
        
        l=1     # first round
        eps_l = eps_frst/4.0
        delta_l = delta_frst/2.0
        
        true_values_l = true_values
        naB_l = naB
        
        runs = 0
        Reward=[]
    while(naB_l!=1):
    
    # each arm of bandits  to be sampled
        smpl_no = math.log10(3.0/delta_l)*4/(eps_l**2) 
        Imm_Reward_l = np.zeros((bandits,naB_l))
        for i in range(int(smpl_no)):
            Imm_Reward = np.random.normal(true_values_l,1) 
            Imm_Reward_l = Imm_Reward_l + Imm_Reward 
            Reward.append(np.mean(Imm_Reward)) 
            runs = runs + 1
        
        Avg_Imm_Reward_l = Imm_Reward_l/int(smpl_no)
        start_median = time.clock()
        Median_l = np.median(Avg_Imm_Reward_l,axis = 1)
        end_median = time.clock()
        times = times + (end_median-start_median)
        true_val_new = np.zeros((bandits,(naB_l-int(naB_l/2))))
        
        for b in range(bandits):
            arm_pos=0
            for arm in range(naB_l):
                if Avg_Imm_Reward_l[b][arm] >= Median_l[b]:
                    true_val_new[b][arm_pos]=true_values_l[b][arm]
                    arm_pos = arm_pos + 1
        true_values_l = true_val_new
        naB_l = naB_l-int(naB_l/2)
        eps_l = 3*eps_l/4.0
        delta_l = delta_l/2
        l=l+1
        
    smpl_no = math.log10(3.0/delta_l)*4/(eps_l**2)
    Imm_Reward_l = np.zeros((bandits,naB_l))
    for i in range(int(smpl_no)):
        Imm_Reward = np.random.normal(true_values_l,1)
        Imm_Reward_l = Imm_Reward_l + Imm_Reward
        Reward.append(np.mean(Imm_Reward))
        runs = runs + 1
            
    end = time.clock()
    totl_time = -start+end
    print('Median_computing_time for ε = %f, δ = %f is %f' %(pairs[0], pairs[1], times))
    print('Total_computing_time for ε = %f, δ = %f is %f' %(pairs[0],pairs[1],totl_time))
#        return runs, Reward, pairs

    
    plt.plot(range(runs), Reward, pairs[2],label = f'ε = {pairs[0]}, δ ={pairs[1]}s',c = pairs[2])
    plt.xlabel('Number of Trials ')
    plt.ylabel('Average Reward')
    plt.title('Median Elimination Algorithm')
    plt.legend()
    
MedianElimination(naB,bandits,true_values,eps_delta_pairs)


end  = time.time()

print("\n \n The total time has taken to run this codes is ",end-start, " seconds")  
  
  
  
  
  
  