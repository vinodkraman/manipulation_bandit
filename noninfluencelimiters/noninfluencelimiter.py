from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy

class NonInfluenceLimiter():
    def __init__(self, bandit, agency, gamma, reward_reports):
        self.bandit = bandit
        self.agency = agency
        self.reward_reports = reward_reports
        self.gamma = gamma
        super().__init__()

    def reset(self):
        self.bandit.reset()
    
    def __compute_NIL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # self.posterior_history[arm_index] = [BetaDistribution(1, 1)]
            pre_alpha, pre_beta = copy.deepcopy(arm.reward_dist.get_params())
            k = 2/(len(self.agency.agents) + 1)
            prev_ema = copy.deepcopy(self.agency.agent_reports[0][arm_index])
        
            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                # print("agent:", agent_index)
                # print("agent reputation:", self.agent_reputations[agent_index])
                gamma = min(1, self.agent_reputations[agent_index])
                current_ema = (self.agency.agent_reports[agent_index][arm_index] - prev_ema) * k + prev_ema
                alpha_j = current_ema * (agent.num_reports) #+ pre_alpha
                beta_j = (1-current_ema) * (agent.num_reports) #+ pre_beta


                q_j = copy.deepcopy(alpha_j/(alpha_j + beta_j))

                alpha_tilde = q_j_tilde * (agent.num_reports) 
                beta_tilde = (1-q_j_tilde) * (agent.num_reports)
    
            # print("final:", alpha_tilde + pre_alpha, beta_tilde + pre_beta)
            arm.influence_reward_dist.set_params(alpha_tilde + pre_alpha,beta_tilde + pre_beta)
    def select_arm(self, t, influence_limit= True):
        self.__compute_NIL_posterior()
        # [print(arm.influence_reward_dist.get_params()) for arm in self.bandit.arms]
        return self.bandit.select_arm(t, influence_limit= influence_limit)

    def __compute_NT_posterior(self, arm, reward):
        self.bandit.arms[arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self.__compute_NT_posterior(arm, reward)