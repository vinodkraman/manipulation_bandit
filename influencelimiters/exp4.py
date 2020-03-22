from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt
from scipy.special import softmax

class Exp4():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        self.gamma = 0.01
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

    def plot_posterior_history(self, arm):
        x = np.linspace(0, 1.0, 100)
        for (index, dist) in enumerate(self.prediction_history[arm]):
            a, b = dist.get_params()
            y = beta.pdf(x, a, b)
            plt.plot(x, y, label=index)
        plt.legend()
        plt.show()

    def normalize_advice(self, advice):
        return softmax(advice)

    def normalize_advice_test(self, advice, tau= 0.5):
        divided = np.divide(advice, tau)
        # print("advice", advice)
        # print("tau", tau)
        divided_exp = np.exp(divided)
        sum_divided_exp = np.sum(divided_exp)
        # print("divided_exp", divided_exp)
        # print("sum_divided_exp", sum_divided_exp)
        return np.divide(divided_exp, sum_divided_exp)
    
    def _compute_IL_posterior(self, t):
        normalized_agent_advice = {}
        self.arm_probs = []
        for agent_index, agent in enumerate(self.agency.agents):
            # print(self.agency.agent_reports[agent])
            norm = self.normalize_advice_test(self.agency.agent_reports[agent])
            normalized_agent_advice[agent] = norm
            # print(norm)

        for (arm_index, arm) in enumerate(self.bandit.arms):
            prediction_weighted_sum = 0
            weight_sum = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                w = self.agent_reputations[agent]
                weight_sum += w
                prediction_weighted_sum += w * normalized_agent_advice[agent][arm_index]

            mean = prediction_weighted_sum/weight_sum
            self.arm_probs.append((1-self.gamma)*mean + self.gamma/self.bandit.K)

    def select_arm(self, t, best_arm=None, influence_limit = True):
        self._compute_IL_posterior(t)
        arm = np.random.choice(self.bandit.K, 1, p=self.arm_probs)[0]
        return arm
        #we should also use quantile for the predictions!

    def _update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            normalized_advice = self.normalize_advice(self.agency.agent_reports[agent])
            y_hat = (reward/self.arm_probs[arm]) * normalized_advice[arm]
            w = self.agent_reputations[agent]
            self.agent_reputations[agent] = w*np.exp(self.gamma * y_hat/self.bandit.K)
            if self.track_reputation == True:
                self.agent_reputations_track[agent].append(self.agent_reputations[agent])

    # def _compute_T_posterior(self, selected_arm, reward):
    #     # self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self._update_reputations(arm, reward)
    
    def plot_reputations(self):
        for (agent, reputations) in self.agent_reputations_track.items():
            plt.plot(reputations, label=agent.id)
        plt.legend()
        plt.xlabel("Round (t)")
        # plt.ylim([0, 1])
        plt.ylabel("Reputation")
        plt.show()

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2

