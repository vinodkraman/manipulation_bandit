from Bandit import Bandit
import numpy as np

class BayesUCB(Bandit):

    def select_arm(self, t, influence_limit= False, c=5):
        selected_arm = 0
        max_value = 0

        #select arm
        for (index, arm) in enumerate(self.arms):
            prob = 1 - (1/(t * np.log(self.T)**(c)))
            val = arm.reward_dist_quantile(prob, influence_limit = influence_limit) 
            if val > max_value:
                max_value = val
                selected_arm = index

        return selected_arm
        
    def update_arm(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)
