from bandits.bandit import Bandit
import numpy as np

class UCB_influence_2(Bandit):

    def select_arm(self, t, q_tilde, W):
        selected_arm = 0
        max_value = 0
        # print(W)
        if t <= len(self.arms):
            return t-1
        else:
            #select arm
            for (index, arm) in enumerate(self.arms):
                # val = arm.mean_reward()*(arm.pulls)/(arm.pulls + W) + q_tilde[index] * (W/(arm.pulls + W)) + np.sqrt((2 * np.log(t)/((arm.pulls) + W)))
                # val = arm.mean_reward()*(1)/(np.exp(W)) + q_tilde[index] * (1 - 1/(np.exp(W))) + np.sqrt((2 * np.log(t)/((arm.pulls) + W)))
                # val = arm.mean_reward()*(1)/(np.exp(W)) + q_tilde[index] * (1 - 1/(np.exp(W)))
                val = (arm.mean_reward())*(1)/(W) + q_tilde[index] * (1 - 1/(W))
                # val = arm.mean_reward() + np.sqrt((2 * np.log(t)/arm.pulls))

                if val > max_value:
                    max_value = val
                    selected_arm = index

            return selected_arm
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
