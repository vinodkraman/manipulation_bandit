from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt

class NonInfluenceLimiter_UCB():
    def __init__(self, bandit, agency, reward_reports, C, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.track_reputation = track_reputation
        self.q_tilde = []
        self.reports = []
        self.C = C
        self.goog = 0
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.prediction_history = {}
    
    def _compute_IL_posterior(self, t):
        self.q_tilde = []
        self.reports = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            weight = 0 
            running_sum = 0
            num_reports = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                running_sum += self.agency.agent_reports[agent][arm_index]
                weight += 1
                num_reports += 1 * agent.num_reports
    
            self.q_tilde.append(running_sum/weight)
            self.reports.append(num_reports)

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        W = 0
        
        for agent in self.agency.agents:
            W += 1
        
        # return self.bandit.select_arm(t, self.q_tilde, W, self.C), 0
        selected_arm = self.bandit.select_arm(t, self.q_tilde, self.reports, W)
        return selected_arm, 0
        #we should also use quantile for the predictions!
    
    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.update(selected_arm, reward)

    def update(self, arm, reward):
        self._compute_T_posterior(arm, reward)
