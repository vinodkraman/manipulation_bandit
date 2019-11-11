from Bandit import Bandit
import numpy as np

class ThompsonSampling(Bandit):
    def select_arm(self, t, influence_limit = False):
        selected_arm = 0
        max_sample = 0

        #select arm
        for (index, arm) in self.arms.items():
            sample_theta = arm.sample(influence_limit = influence_limit)
            if sample_theta > max_sample:
                max_sample = sample_theta
                selected_arm = index

        return selected_arm
        
    def update_arm(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)