from bandits.bandit import Bandit
import numpy as np

class BayesUCB_test(Bandit):

    def select_arm(self, t, q_tilde, W, C, order= None, influence_limit= False, c=5): 
        if t <= 1*len(self.arms):
            return (t-1)%self.K, (t-1)%self.K

        #select arm
        val_arm_pulls = [arm.real_pulls for (index, arm) in enumerate(self.arms)]
        filtered_vals = self.filter_arms(val_arm_pulls, t)

        prob = 1 - (1/(t * np.log(self.T)**(c)))
        val_ucb = [arm.reward_dist_quantile(prob, influence_limit = influence_limit)  for arm in self.arms]

        means = [val_ucb[index] for index in filtered_vals]
        index_index = np.argmax(means)
        return filtered_vals[index_index], np.argmax(q_tilde)
        
    def filter_arms(self, pulls, t):
        filtered_indices = []
        sorted_pulls = np.argsort(pulls)
        total_sum = np.sum(pulls)
        temp_sum = np.sum(pulls)
        # 1/np.log(.001*t+np.exp(1))
        for index in sorted_pulls:
            if (temp_sum - pulls[index])/total_sum >= 0.97:
                temp_sum -= pulls[index]
            else:
                filtered_indices.append(index)

        return filtered_indices

    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)
