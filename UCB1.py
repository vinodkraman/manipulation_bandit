from Bandit import Bandit
import numpy as np

class UCB1(Bandit):

    def select_arm(self, t):
        selected_arm = 0
        max_value = 0

        #select arm
        for (index, arm) in self.arms.items():
            val = arm.mean_reward() + np.sqrt((2 * np.log(t)/arm.pulls))
            if val > max_value:
                max_value = val
                selected_arm = index

        return selected_arm
        
    def update_arm(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
