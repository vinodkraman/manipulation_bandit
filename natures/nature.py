import random
from agencies.agency import Agency
from scipy.stats import bernoulli
from distributions.bernoullidistribution import BernoulliDistribution
from distributions.betadistribution import BetaDistribution
import numpy as np

class Nature():
    def __init__(self, num_arms, world_priors, num_agents= 5):
        self.num_agents = num_agents
        self.num_arms = num_arms 
        self.arm_dists = []
        self.best_arm_mean = 0
        self.best_arm = 0
        self.agency = Agency()
        self.world_priors = world_priors


    def reset_world_prior(self):
        vec = np.random.uniform(0, 1, self.num_arms)
        self.world_priors = [BetaDistribution(5*vec[k], 5*(1-vec[k])) for k in range(self.num_arms)]

    def initialize_arms(self):
        self.hidden_params = [prior.sample() for prior in self.world_priors]
        # self.hidden_params = [0.5, 0.3, 0.9, 0.2, 0.7]
        # print(self.hidden_params)
        self.arm_dists = [BernoulliDistribution(param) for param in self.hidden_params]
        self.best_arm_mean = max(self.hidden_params)
        self.best_arm = np.argmax(self.hidden_params)
        self.worst_arm = np.argmin(self.hidden_params)

    def reinitialize_arms(self):
        self.hidden_params = [prior.sample() for prior in self.world_priors]
        self.arm_dists = [BernoulliDistribution(param) for param in self.hidden_params]
        self.best_arm_mean = max(self.hidden_params)
        self.best_arm = np.argmax(self.hidden_params)
        self.worst_arm = np.argmin(self.hidden_params)

        target_arm = np.argmin(self.hidden_params)
        # target_arms = set(target_arm)

        for agent in self.agency.agents:
            agent.arm_dists = self.arm_dists #a array of distributions
            agent.best_arm = self.best_arm
            agent.target_arms = {target_arm}
            agent.worst_arm = self.worst_arm

    # def initialize_agents(self, trustworthy, num_reports):
    #     self.agency.clear_agents()
    #     for i in range(self.num_agents):
    #         if trustworthy[i]:
    #             self.agency.create_agent(trustworthy[i], self.arm_dists,num_reports)
    #         else:
    #             self.agency.create_agent(trustworthy[i], self.malicious_dists, num_reports)

    def initialize_agents(self, trustworthy, num_reports, num_target_items, attack_freq):
        self.agency.clear_agents()
        target_arm = np.argmin(self.hidden_params)
        options = [x for x in range(self.num_arms) if (x != self.best_arm and x != target_arm)]
        target_arms = random.sample(options, num_target_items-1)
        target_arms.append(target_arm)
        target_arms = set(target_arms)
        # print("target_arm", target_arms)
        # print("target_arm ", target_arm)
        for i in range(self.num_agents):
            self.agency.create_agent(trustworthy[i], self.arm_dists, num_reports, self.best_arm, target_arms, attack_freq, i)

    def shuffle_agents(self):
        self.agency.shuffle_agents()


    def get_agent_reports(self, t, attack= "copy"):
        return self.agency.send_reports(t, attack)

    def generate_reward(self, arm):
        return self.arm_dists[arm].sample()
    
    def generate_rewards(self):
        self.rewards = []
        for dist in self.arm_dists:
            self.rewards.append(dist.sample())

        return self.rewards

    def compute_per_round_regret(self, arm, reward=None):
        if reward == None:
            return self.best_arm_mean - self.hidden_params[arm]
        else:
            return self.generate_reward(self.best_arm) - reward

    def compute_per_round_trust_regret(self, arm, oracle_arm):
        return max(self.hidden_params[oracle_arm] - self.hidden_params[arm],0)

    def compute_per_round_new_regret(self, arm, prediction, oracle_arm, oracle_prediction, rewards):
        # print(prediction, oracle_prediction, rewards[arm])
        # losses = []
        # for index, arm_prediction in enumerate(oracle_prediction):
        #     losses.append(self.scoring_rule(rewards[index], arm_prediction))

        # return self.scoring_rule(rewards[arm], prediction) - np.dot(losses, oracle_prob)
        return self.scoring_rule(rewards[arm], prediction) - self.scoring_rule(rewards[oracle_arm], oracle_prediction)


    # def compute_per_round_best_expert_regret(self, arm, prediction, best_expert_prediction, rewards):
        


    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2
