from Bandit import Bandit
from BetaDistribution import BetaDistribution
import numpy as np
import copy

class NonInfluenceLimiter():
    def __init__(self, bandit, agency, reward_reports):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.reward_reports = reward_reports
        super().__init__()

    def compute_IL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            for index, agent in enumerate(self.agency.agents):
                alpha_tilde += self.agency.agent_reports[index][arm_index]*agent.num_reports
                beta_tilde += (1-self.agency.agent_reports[index][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, t, influence_limit = True):
       return self.bandit.select_arm(t, influence_limit = influence_limit)

    def compute_T_posterior(self, arm, reward):
        reward_alpha = (reward == 1) * self.reward_reports
        reward_beta = (reward == 0) * self.reward_reports
        self.bandit.arms[arm].reward_dist.update(reward_alpha, reward_beta)


    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2