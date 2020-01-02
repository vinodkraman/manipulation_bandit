from Bandit import Bandit
from BetaDistribution import BetaDistribution
import numpy as np
import copy

class NonInfluenceLimiter4():
    def __init__(self, bandit, agency, gamma, reward_reports):
        self.bandit = bandit
        self.agency = agency
        self.reward_reports = reward_reports
        self.gamma = gamma
        super().__init__()

    def reset(self):
        self.bandit.reset()

    # def __compute_NIL_posterior(self):
    #     # print("reports:", self.agency.agent_reports)
    #     for (arm_index, arm) in enumerate(self.bandit.arms):
    #         alpha_tilde, beta_tilde = arm.reward_dist.get_params()
    #         # print("arm", arm_index)
    #         # print("prior", alpha_tilde, beta_tilde)

    #         #iterate through each agent and process their report
    #         for index, agent in enumerate(self.agency.agents):
    #             alpha_tilde += self.agency.agent_reports[index][arm_index]*agent.num_reports
    #             beta_tilde += (1-self.agency.agent_reports[index][arm_index])*agent.num_reports

    #         # print("posterior", alpha_tilde, beta_tilde)
    #         arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
    #         #compute posterior and set to bandit influence-limited posterior
    
    def __compute_NIL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            gamma = self.gamma
            for agent_index, agent in enumerate(self.agency.agents):
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[agent_index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[agent_index][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, t, influence_limit= True):
        self.__compute_NIL_posterior()
        # [print(arm.influence_reward_dist.get_params()) for arm in self.bandit.arms]
        return self.bandit.select_arm(t, influence_limit= influence_limit)

    def _compute_NT_posterior(self, selected_arm, reward):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()
            # alpha_tilde = 0
            # beta_tilde = 0
            gamma = self.gamma
            for index, agent in enumerate(self.agency.agents):
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports

            if arm_index == selected_arm:
                alpha_tilde += (reward == 1) * self.reward_reports
                beta_tilde += (reward == 0) * self.reward_reports
            # self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
            arm.reward_dist.set_params(alpha_tilde, beta_tilde)
        # self.bandit.arms[selected_arm].reward_dist.update(reward)


    def update(self, arm, reward):
        self._compute_NT_posterior(arm, reward)