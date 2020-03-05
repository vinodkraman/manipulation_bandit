from distributions.betadistribution import BetaDistribution
import copy
class BanditArm():
    #need three distributions:
        #1) only keep track of rewards and pulls
        #2) keep track of influence-limited posteriors
        #3) keep track of temporary posteriors
    def __init__(self, reward_dist):
        self.reward_dist = reward_dist
        self.influence_reward_dist = copy.deepcopy(reward_dist)
        self.pulls = 0
        self.rewards = 0

    def mean_reward(self):
        return self.rewards/(self.pulls + 1e-5)
    
    def quantile(self, c):
        return self.influence_reward_dist

    def sample(self, influence_limit = False):
        if influence_limit:
            return self.influence_reward_dist.sample()
        else:
            return self.reward_dist.sample()

    def update_reward_dist(self, reward, influence_limit = False):
        if influence_limit:
            print("Hello")
            # self.influence_reward_dist.update(reward == 1, reward == 0)
            self.influence_reward_dist.update(reward)
        else:
            self.reward_dist.update(reward)

    def reward_dist_mean(self, influence_limit = False):
        if influence_limit:
            return self.influence_reward_dist.mean()
        else:
            return self.reward_dist.mean()
    
    def reward_dist_quantile(self, prob, influence_limit=False):
        if influence_limit:
            # print(self.influence_reward_dist.get_params())
            return self.influence_reward_dist.get_quantile(prob)
        else:
            return self.reward_dist.get_quantile(prob)

