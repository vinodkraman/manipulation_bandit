from Bandit import Bandit
from BetaDistribution import BetaDistribution
import numpy as np
import copy

class NonInfluenceLimiter3():
    def __init__(self, bandit, agency, reward_reports):
        self.bandit = bandit
        self.agency = agency
        self.reward_reports = reward_reports
        super().__init__()

    def reset(self):
        self.bandit.reset()

    def __compute_NIL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                gamma = 1
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[agent_index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[agent_index][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, t, influence_limit= True):
        self.__compute_NIL_posterior()
        # [print(arm.influence_reward_dist.get_params()) for arm in self.bandit.arms]
        return self.bandit.select_arm(t, influence_limit= influence_limit)

    def __compute_NT_posterior(self, arm, reward):
        self.bandit.arms[arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self.__compute_NT_posterior(arm, reward)