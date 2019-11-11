from random import *
from Agency import Agency
class Nature():
    def __init__(self, num_agents, num_arms, num_samples, bandits, manipulation_strategy = False):
        self.num_agents = num_agents
        self.manipulation_strategy = manipulation_strategy
        self.num_arms = num_arms #a array of distributions
        self.num_samples = num_samples
        self.arm_distributions = []
        self.best_arm = 0
        self.bandits = bandits
        self.agency = Agency()

    def initialize_arms(self):
        self.arm_distributions = [random() for i in range(K)]
        self.best_arm_mean = max(self.arm_distributions)
            




#truthful to one arm, malicious to another?