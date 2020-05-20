from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy

class Oracle_ucb():
    def __init__(self, bandit, agency, C):
        self.bandit = bandit
        self.agency = agency
        self.q_tilde = []
        self.C = C
        self.goog = 0
        self.reports = []
        super().__init__()

    def reset(self):
        self.bandit.reset()

    def __compute_trust_posterior(self, t):
        self.q_tilde = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # arm.influence_reward_dist = copy.deepcopy(arm.reward_dist)
            initial = 0
            if t <= self.bandit.K:
                initial = 0.5
            else:
                initial = 0.5
            reports = [initial]
            reports = []
            num_reports = 0
            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                if agent.trustworthy == True:
                    num_reports += 1 * agent.num_reports
                    reports.append(self.agency.agent_reports[agent][arm_index])
                    

            self.q_tilde.append(np.mean(reports))
            self.reports.append(num_reports)

    def select_arm(self, t, influence_limit = True):
        self.__compute_trust_posterior(t)
        # print("predictions", self.q_tilde)
        W = 0
        
        for agent_index, agent in enumerate(self.agency.agents):
            if agent.trustworthy == True:
                 W += 1

        selected_arm = self.bandit.select_arm(t, self.q_tilde, self.reports, W)
        return selected_arm, 0
        # selected_arm, prediction = self.bandit.select_arm(t, self.q_tilde, W, self.C)
        # self.goog = prediction
        # return selected_arm, 0
        # return self.bandit.select_arm(t, self.q_tilde, W, self.C), 0

    def update(self, arm, reward):
        # self.bandit.update(arm, reward)
        self.bandit.arms[self.goog].real_pulls += 1
        self.bandit.update(arm, reward)

        # self._compute_T_posterior(arm, reward)

        # self.bandit.arms[arm].real_pulls += 1
        # # print("STATUS:", self.goog)
        # should_explore = np.random.rand()
        # if should_explore < self.goog or self.goog == -1:
        # # if self.goog == False:
        #     # self._update_reputations(arm, reward)
        #     self.bandit.update(arm, reward)
