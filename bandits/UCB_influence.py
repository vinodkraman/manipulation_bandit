from bandits.bandit import Bandit
import numpy as np

class UCB_influence(Bandit):

    def select_arm(self, t, q_tilde, W):
        selected_arm = 0
        max_value = 0

        #select arm
        for (index, arm) in enumerate(self.arms):
            val = arm.mean_reward()*(np.log(arm.pulls + 1e-5)/(np.log(arm.pulls + 1e-5) + W)) + q_tilde[index] * (W/(np.log(arm.pulls + 1e-5) + W)) + np.sqrt((2 * np.log(t)/((arm.pulls+1e-5))))
            if val > max_value:
                max_value = val
                selected_arm = index

        return selected_arm
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
