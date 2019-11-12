from random import *
from Agency import Agency
from scipy.stats import bernoulli
from BernoulliDistribution import BernoulliDistribution

class Nature():
    def __init__(self, num_agents, num_arms):
        self.num_agents = num_agents
        self.num_arms = num_arms 
        self.arm_dists = []
        self.malicious_dists = []
        self.best_arm = 0
        self.agency = Agency()

    def initialize_arms(self):
        self.hidden_theta = [random() for i in range(self.num_arms)]
        self.arm_dists = [BernoulliDistribution(theta) for theta in self.hidden_theta]
        self.malicious_dists = [BernoulliDistribution(1-theta) for theta in self.hidden_theta]
        self.best_arm_mean = max(self.hidden_theta)

    def initialize_agents(self, malicious, num_reports, initial_reputation):
        for i in range(self.num_agents):
            if malicious[i]:
                self.agency.create_agent(malicious[i], self.malicious_dists,num_reports, initial_reputation)
            else:
                self.agency.create_agent(malicious[i], self.arm_dists,num_reports, initial_reputation)

    def get_agent_reports(self):
        return self.agency.send_reports()

    def generate_reward(self, arm):
        return self.arm_dists[arm].sample()