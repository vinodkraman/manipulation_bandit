from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy

class Oracle_exp():
    def __init__(self, bandit, agency):
        self.bandit = bandit
        self.agency = agency
        self.q_tilde = []
        super().__init__()

    def reset(self):
        self.bandit.reset()

    def __compute_trust_posterior(self):
        self.q_tilde = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # arm.influence_reward_dist = copy.deepcopy(arm.reward_dist)
            reports = []
            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                if agent.trustworthy == True:
                    reports.append(self.agency.agent_reports[agent][arm_index])
                    

            self.q_tilde.append(np.mean(reports))

    def select_arm(self, t, influence_limit = True):
        self.__compute_trust_posterior()
        arm = np.argmax(self.q_tilde)
        return arm, self.q_tilde[arm]

    def update(self, arm, reward):
        pass