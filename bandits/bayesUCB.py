from bandits.bandit import Bandit
import numpy as np

class BayesUCB(Bandit):

    def select_arm(self, t, W=0, influence_limit= False, c=5):
        if t <= 1*len(self.arms):
            return (t-1)%self.K, 0
        # selected_arm = 0
        # max_value = 0

        # #select arm
        # # print("influence_limit", influence_limit)
        # # print(1 - (1/(t * np.log(self.T)**(c))))
        # for (index, arm) in enumerate(self.arms):
        #     prob = 1 - (1/(t * np.log(self.T)**(c)))
        #     val = arm.reward_dist_quantile(prob, influence_limit = influence_limit) 
        #     # print("arm", index)
        #     # print("val", val)
        #     if val > max_value:
        #         max_value = val
        #         selected_arm = index

        # return selected_arm
        # pure_mean_estimate = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        prob = 1 - (1/(t * np.log(self.T)**(c)))
        val_ucb = [arm.reward_dist_quantile(prob, influence_limit = influence_limit)  for arm in self.arms]

        if W <= 1:
            return np.argmax(val_ucb), 0
        else:
            pure_mean_estimate = [arm.reward_dist_mean() for (index, arm) in enumerate(self.arms)]
            return np.argmax(pure_mean_estimate), 0
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)
