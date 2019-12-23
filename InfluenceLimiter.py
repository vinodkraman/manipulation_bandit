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
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        
    def __compute_IL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
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
    def select_arm(self, t, influence_limit = True):
        self.__compute_IL_posterior()
        return self.bandit.select_arm(t, influence_limit= influence_limit)

    def __update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            gamma = min(1, agent.reputation)
            q_tile_j_1 = self.posterior_history[arm][index].mean()
            alpha_delta = 0
            beta_delta = 0
            for i in range(index + 1):
                alpha_delta += self.agency.agent_reports[i][arm] * agent.num_reports
                beta_delta += (1-self.agency.agent_reports[i][arm]) * agent.num_reports

            prev_alpha, prev_beta = self.bandit.arms[arm].reward_dist.get_params()
            q_j = BetaDistribution(prev_alpha + alpha_delta, prev_beta + beta_delta).mean()

            agent.reputation += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))

    # def compute_T_posterior(self, arm, reward):
    #         alpha_tilde, beta_tilde = self.bandit.arms[arm].reward_dist.get_params()
    #         # alpha_tilde = 0
    #         # beta_tilde = 0
    #         for index, agent in enumerate(self.agency.agents):
    #             gamma = min(1, agent.reputation)
    #             alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm]*agent.num_reports
    #             beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm])*agent.num_reports

    #         reward_alpha = (reward == 1) * self.reward_reports
    #         reward_beta = (reward == 0) * self.reward_reports
    #         # self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
    #         self.bandit.arms[arm].reward_dist.set_params(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
    def __compute_T_posterior(self, selected_arm, reward):
        # for (arm_index, arm) in enumerate(self.bandit.arms):
        #     alpha_tilde, beta_tilde = arm.reward_dist.get_params()
        #     # alpha_tilde = 0
        #     # beta_tilde = 0
        #     for index, agent in enumerate(self.agency.agents):
        #         gamma = min(1, agent.reputation)
        #         alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
        #         beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports

        #     if arm_index == selected_arm:
        #         alpha_tilde += (reward == 1) * self.reward_reports
        #         beta_tilde += (reward == 0) * self.reward_reports
        #     # self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
        #     arm.reward_dist.set_params(alpha_tilde, beta_tilde)
        self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self.__update_reputations(arm, reward)
        self.__compute_T_posterior(arm, reward)

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2