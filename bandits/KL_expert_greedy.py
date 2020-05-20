from bandits.bandit import Bandit
import numpy as np
from random import *
from scipy.stats import gumbel_r

class KL_Expert_Greedy(Bandit):

    def compute_KLD(self, p, q):
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        else: 
            # print(q)
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def select_arm(self, t, q_tilde, reports, W):
        should_explore = np.random.rand()
        mean_estimate = [(arm.rewards + q_tilde[index]*reports[index])/(arm.pulls + reports[index]) for (index, arm) in enumerate(self.arms)]

        if should_explore < min(0.10, 1/np.exp(W)):
            return randint(0, self.K-1)
        else:
            return np.argmax(mean_estimate)

    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
