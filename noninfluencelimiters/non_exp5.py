from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt
from scipy.special import softmax

class Non_exp5():
    def __init__(self, bandit, agency, reward_reports, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.track_reputation = track_reputation
        self.temperature = 0.1
        super().__init__()
    
    def reset(self):
        self.bandit.reset()

    def normalize_advice_test(self, advice, tau= 1):
        divided = np.divide(advice, tau)
        divided_exp = np.exp(divided)
        sum_divided_exp = np.sum(divided_exp)
        return np.divide(divided_exp, sum_divided_exp)
    
    def _compute_IL_posterior(self, t):
        self.q_tilde = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            weight = 0  
            running_sum = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                running_sum += self.agency.agent_reports[agent][arm_index]
                weight += 1
    
           
            self.q_tilde.append(running_sum/weight)

    def select_arm(self, t, best_arm= None, influence_limit = True):
        self._compute_IL_posterior(t)

        W = 0
        for agent, reputation in enumerate(self.agency.agents):
            W += 1

        ratio = 1/(W)
        norm_advice = self.normalize_advice_test(self.q_tilde, ratio)
        self.saved_prob = norm_advice

        arm = np.random.choice(self.bandit.K, 1, p=self.saved_prob)[0]

        return arm

    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.update(selected_arm, reward)

    def update(self, arm, reward):
        self._compute_T_posterior(arm, reward)

