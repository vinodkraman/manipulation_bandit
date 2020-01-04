from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy

class Oracle():
    def __init__(self, bandit, agency):
        self.bandit = bandit
        self.agency = agency
        super().__init__()

    def reset(self):
        self.bandit.reset()
        
    def _compute_IL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # arm.influence_reward_dist = copy.deepcopy(arm.reward_dist)
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            gamma = 0
            for agent_index, agent in enumerate(self.agency.agents):
                if agent.trustworthy == True:
                    gamma = 1
                else:
                    gamma = 0

                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[agent_index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[agent_index][arm_index])*agent.num_reports

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)

    def select_arm(self, t, influence_limit= True):
        self._compute_IL_posterior()
        return self.bandit.select_arm(t, influence_limit= influence_limit)
    
    def __compute_posterior(self, selected_arm, reward):
        self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self.__compute_posterior(arm, reward)