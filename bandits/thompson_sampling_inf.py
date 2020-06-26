from bandits.bandit import Bandit
import numpy as np
class Thompson_Sampling_Inf(Bandit):
    def select_arm(self, t, W=0, influence_limit = False):
        # if t <= 2*len(self.arms):
        #     return (t-1)%self.K, 0
        # sampled_val = [arm.sample(influence_limit = influence_limit) for arm in self.arms]
        # return np.argmax(sampled_val), 0
        # for arm in
        # print(arm.reward_dist.get_params())
        # print(arm.influence_reward_dist.get_params())
        upper_quantiles = [arm.reward_dist_quantile(0.99, influence_limit = False)  for arm in self.arms]
        lower_quantiles = [arm.reward_dist_quantile(0.01, influence_limit = False)  for arm in self.arms]
        # print(upper_quantiles)
        # print(lower_quantiles)

        sampled_val_inf = [arm.sample(influence_limit = True) for arm in self.arms]

        clipped = np.clip(sampled_val_inf, lower_quantiles, upper_quantiles)
        return np.argmax(clipped), 0


        inf_index = np.argmax(sampled_val_inf)
        sampled_val = [arm.sample(influence_limit = False) for arm in self.arms]
        index = np.argmax(sampled_val)

        # should_explore = 1 - 1/t

        # if np.random.rand() > should_explore:
        #     return inf_index, 0
        # else:
        #     return index, 0
        # frac = 0.25
        # final = [frac*sampled_val_inf[index] + (1-frac)*sampled_val[index] for index in range(self.K)]



        # if sampled_val_inf[inf_index] > sampled_val[index]:
        #     return inf_index, 0
        # else:
        #     return index, 0

        final = [max(sampled_val[index], sampled_val_inf[index]) for index in range(self.K)]
        return np.argmax(final), 0
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
        self.arms[arm].update_reward_dist(reward)