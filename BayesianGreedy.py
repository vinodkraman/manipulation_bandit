from Bandit import Bandit
import numpy as np
from random import *

class BayesianGreedy(Bandit):
    def select_arm(self, t, influence_limit = False):
        should_explore = random() > self.epsilon
        if should_explore:
            return randint(0, self.K-1)
        else:
            max_value = 0
            selected_arm = 0
            for (index, arm) in self.arms.items():
                val = arm.reward_dist_mean(influence_limit = influence_limit)
                if val > max_value:
                    max_value = val
                    selected_arm = index
            return selected_arm
        
    def update_arm(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)