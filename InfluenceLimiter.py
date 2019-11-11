from Bandit import Bandit
from BetaDistribution import BetaDistribution
import numpy as np

class InfluenceLimiter():
    def __init__(self, bandit, agency, rule):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.rule = rule
        super().__init__()

    def compute_IL_posterior(self):
        for arm in self.bandit.arms:
            arm.influence_reward_dist = arm.reward_dist
            self.posterior_history[arm] = [arm.reward_dist]

            alpha_tilde = 0
            beta_tilde = 0
            #iterate through each agent and process their report
            for index, agent in self.agency.agents:
                gamma = min(1, agent.reputation)
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm])*agent.num_reports
                self.posterior_history[arm].append(BetaDistribution(alpha_tilde, beta_tilde))

            arm.influence_reward_dist.update(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, t):
       return self.bandit.select_arm(influence_limit = True)

    def update_reputations(self, arm, reward):
        for index, agent in self.agency.agents:
            gamma = min(1, agent.reputation)
            q_tile_j_1 = self.posterior_history[arm][index].mean()
            alpha_delta = 0
            beta_delta = 0
            for i in range(index + 1):
                alpha_delta += self.agency.agent_reports[index][arm] * agent.num_reports
                beta_delta += (1-self.agency.agent_reports[index][arm]) * agent.num_reports

            prev_alpha, prev_beta = arm.reward_dist.get_params()
            q_j = BetaDistribution(prev_alpha + alpha_delta, prev_beta + beta_delta).mean()

            agent.reputation += agent.reputation + gamma * (self.scoring_rule(reward,q_tile_j_1) - self.scoring_rule(reward,q_j))

    def compute_T_posterior(self, arm, reward):
            alpha_tilde = 0
            beta_tilde = 0
            for index, agent in self.agency.agents:
                gamma = min(1, agent.reputation)
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm])*agent.num_reports

            reward_alpha = (reward == 1) * self.agency.num_reports
            reward_beta = (reward == 0) * self.agency.num_reports
            arm.reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return q**2
        else:
            return (1-q)**2