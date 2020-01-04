from Bandits.Bandit import Bandit
from Distributions.BetaDistribution import BetaDistribution
import numpy as np
import copy

class Oracle():
    def __init__(self, bandit, agency):
        self.bandit = bandit
        self.agency = agency
        super().__init__()

    def reset(self):
        self.bandit.reset()

    def __compute_trust_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            for index, agent in enumerate(self.agency.agents):
                if agent.trustworthy == True:
                    alpha_tilde += self.agency.agent_reports[index][arm_index]*agent.num_reports
                    beta_tilde += (1-self.agency.agent_reports[index][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior

    def select_arm(self, t, influence_limit = True):
        self.__compute_trust_posterior()
        return self.bandit.select_arm(t, influence_limit= influence_limit)
    
    def __compute_posterior(self, arm, reward):
        self.bandit.arms[arm].reward_dist.update(reward)
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.influence_reward_dist.get_params()
            arm.reward_dist.set_params(alpha_tilde, beta_tilde)

    def update(self, arm, reward):
        self.__compute_posterior(arm, reward)