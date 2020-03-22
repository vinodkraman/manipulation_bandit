from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt

class NonInfluenceLimiter_UCB():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        self.q_tilde = {}
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.prediction_history = {}
        self.__initialize_reputations()

    def __initialize_reputations(self):
        self.agent_reputations = {agent:self.initial_reputation for agent in self.agency.agents}
        # self.agent_reputations = [int(agent.trustworthy == True) for agent in self.agency.agents]
        if self.track_reputation:
            self.agent_reputations_track = {agent:[self.initial_reputation] for agent in self.agency.agents}
    
    def _compute_IL_posterior(self, t):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            weight = 0  
            running_sum = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                running_sum += self.agency.agent_reports[agent][arm_index]
                weight += 1
    
           
            self.q_tilde[arm_index] = running_sum/weight

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        W = 0
        
        for agent, reputation in self.agent_reputations.items():
            W += reputation
        
        return self.bandit.select_arm(t, self.q_tilde, W)
        #we should also use quantile for the predictions!
    
    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.update(selected_arm, reward)

    def update(self, arm, reward):
        self._compute_T_posterior(arm, reward)

