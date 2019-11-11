from Bandit import Bandit
from random import *
import numpy as np

class EpsilonGreedy(Bandit):
    def select_arm(self, t, influence_limit = False):
        should_explore = random() > self.epsilon

        if should_explore:
            return randint(0, self.K-1)
        else:
            max_value = 0
            selected_arm = 0
            for (index, arm) in self.arms.items():
                val = arm.mean_reward()
                if val > max_value:
                    max_value = val
                    selected_arm = index
            return selected_arm
        
    def update_arm(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
