from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy

class Oracle2():
    def __init__(self, bandit, agency):
        self.bandit = bandit
        self.agency = agency
        super().__init__()

    def reset(self):
        self.bandit.reset()

    def __compute_trust_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # arm.influence_reward_dist = copy.deepcopy(arm.reward_dist)
            alpha_tilde, beta_tilde = 0, 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                if agent.trustworthy == True:
                    alpha_tilde += self.agency.agent_reports[agent][arm_index]*agent.num_reports
                    beta_tilde += (1-self.agency.agent_reports[agent][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)

    def select_arm(self, t, influence_limit = True):
        self.__compute_trust_posterior()
        return self.bandit.select_arm(t, influence_limit= influence_limit)

    def update(self, arm, reward):
        pass