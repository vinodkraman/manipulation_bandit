from bandits.bandit import Bandit
import numpy as np
from random import *
from scipy.stats import expon
from scipy.stats import binom

class UCB_Expert_Greedy(Bandit):

    def compute_KLD(self, p, q):
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        else: 
            # print(q)
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def select_arm(self, t, q_tilde, W, C):
        if t <= len(self.arms):
            return t-1
        # should_explore = np.random.rand()
        # if should_explore < .30:
        #     val = [arm.mean_reward() for arm in self.arms]
        #     return np.argmax(val)
        # else:
        #     return np.argmax(q_tilde)

        if W < 1:
            val = [arm.mean_reward() + np.sqrt(2*np.log(t)/arm.pulls) for arm in self.arms]
            return np.argmax(val)
            # return randint(0, self.K-1)
        else:
            # should_explore = np.random.rand()
            # if should_explore < .50:
            #     val = [arm.mean_reward() for arm in self.arms]
            #     return np.argmax(val)
            # else:
            # val = [max(q_tilde[index], arm.mean_reward()) for index, arm in enumerate(self.arms)]
            return np.argmax(q_tilde)
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
