from Bandit import Bandit
from BetaDistribution import BetaDistribution
import numpy as np
import copy

class InfluenceLimiter():
    def __init__(self, bandit, agency, reward_reports):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.reward_reports = reward_reports
        super().__init__()

    def compute_IL_posterior(self):
        for (arm_index, arm) in self.bandit.arms.items():
            arm.influence_reward_dist = copy.copy(arm.reward_dist)
            self.posterior_history[arm_index] = [copy.copy(arm.reward_dist)]

            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            for index, agent in enumerate(self.agency.agents):
                gamma = min(1, agent.reputation)
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports
                self.posterior_history[arm_index].append(BetaDistribution(alpha_tilde, beta_tilde))

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, influence_limit = True):
       return self.bandit.select_arm(1, influence_limit = influence_limit)

    def update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            print("agent:", index)
            gamma = min(1, agent.reputation)
            q_tile_j_1 = self.posterior_history[arm][index].mean()
            print("q_tilde_j_1:", q_tile_j_1)
            alpha_delta = 0
            beta_delta = 0
            for i in range(index + 1):
                alpha_delta += self.agency.agent_reports[i][arm] * agent.num_reports
                beta_delta += (1-self.agency.agent_reports[i][arm]) * agent.num_reports

            prev_alpha, prev_beta = self.bandit.arms[arm].reward_dist.get_params()
            q_j = BetaDistribution(prev_alpha + alpha_delta, prev_beta + beta_delta).mean()
            print("q_j:", q_j)

            agent.reputation += agent.reputation + gamma * (self.scoring_rule(reward,q_tile_j_1) - self.scoring_rule(reward,q_j))

    def compute_T_posterior(self, arm, reward):
            alpha_tilde = 0
            beta_tilde = 0
            for index, agent in enumerate(self.agency.agents):
                gamma = min(1, agent.reputation)
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm])*agent.num_reports

            reward_alpha = (reward == 1) * self.reward_reports
            reward_beta = (reward == 0) * self.reward_reports
            self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return q**2
        else:
            return (1-q)**2