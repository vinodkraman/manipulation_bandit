from bandits.bandit import Bandit
import numpy as np

class Thompson_Sampling(Bandit):
    def select_arm(self, t, W=0, influence_limit = False):
        sampled_val = [arm.sample(influence_limit = influence_limit) for arm in self.arms]
        return np.argmax(sampled_val), 0

        # #select arm
        # for (index, arm) in enumerate(self.arms):
        #     sample_theta = arm.sample(influence_limit = influence_limit)
        #     if sample_theta > max_sample:
        #         max_sample = sample_theta
        #         selected_arm = index

        # return selected_arm
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)


#with some probability, you exploit your current knowledge, and with some probability
#you exploit expert advice you cant know when is right and wrong. So your best shot
# is to probabilistically just check. 