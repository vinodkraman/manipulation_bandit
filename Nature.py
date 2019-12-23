from random import *
from Agency import Agency
from scipy.stats import bernoulli
from BernoulliDistribution import BernoulliDistribution

class Nature():
    def __init__(self, num_arms, world_priors, num_agents= 5):
        self.num_agents = num_agents
        self.num_arms = num_arms 
        self.arm_dists = []
        self.malicious_dists = []
        self.best_arm_mean = 0
        self.agency = Agency()
        self.world_priors = world_priors

    def initialize_arms(self):
        self.hidden_params = [prior.sample() for prior in self.world_priors]
        # print(self.hidden_params)
        self.arm_dists = [BernoulliDistribution(param) for param in self.hidden_params]
        self.malicious_dists = [BernoulliDistribution(1-param) for param in self.hidden_params]
        self.best_arm_mean = max(self.hidden_params)

    def initialize_agents(self, trustworthy, num_reports, initial_reputation):
        self.agency.clear_agents()
        for i in range(self.num_agents):
            if trustworthy[i]:
                self.agency.create_agent(trustworthy[i], self.arm_dists,num_reports, initial_reputation)
            else:
                self.agency.create_agent(trustworthy[i], self.malicious_dists, num_reports, initial_reputation)

    def get_agent_reports(self):
        return self.agency.send_reports()

    def generate_reward(self, arm):
        return self.arm_dists[arm].sample()

    def compute_per_round_regret(self, arm):
        return self.best_arm_mean - self.hidden_params[arm]

    # def compute_per_round_trust_regret(self, arm):
