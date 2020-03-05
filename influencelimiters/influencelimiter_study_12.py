from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt

class InfluenceLimiter_study_12():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        self.agent_loss = {}
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.prediction_history = {}
        self.__initialize_reputations()

    def __initialize_reputations(self):
        self.agent_loss = {agent:0 for agent in self.agency.agents}

    def plot_posterior_history(self, arm):
        x = np.linspace(0, 1.0, 100)
        for (index, dist) in enumerate(self.prediction_history[arm]):
            a, b = dist.get_params()
            y = beta.pdf(x, a, b)
            plt.plot(x, y, label=index)
        plt.legend()
        plt.show()

    def _compute_SMA(self, arm_index):
        npa = np.asarray(copy.deepcopy(self.agency.agent_reports))
        return np.mean(npa[:,arm_index])
    
    def _compute_IL_posterior(self, t):
        eta = np.sqrt((8 * np.log(len(self.agency.agents)))/self.bandit.T)
        for (arm_index, arm) in enumerate(self.bandit.arms):
            pre_alpha, pre_beta = copy.deepcopy(arm.reward_dist.get_params())

            #iterate through each agent and process their report
            leader = None
            smallest_loss = np.inf
            for agent_index, agent in enumerate(self.agency.agents):
                loss = self.agent_loss[agent] + (eta)*self.agency.agent_reports[agent][arm_index] * np.log(self.agency.agent_reports[agent][arm_index]+1e-5)
                if loss <= smallest_loss:
                    smallest_loss = loss
                    leader = agent
            
            # print("arm, leader:", arm_index, leader.id)
            # [print(agent.id, loss) for agent,loss in self.agent_loss.items()]
            final_pred = self.agency.agent_reports[leader][arm_index]
            alpha_tilde = final_pred * leader.num_reports
            beta_tilde = (1-final_pred) * leader.num_reports
            # y_tilde_alpha = (prediction_weighted_sum/weight_sum) * (reports)
            # y_tilde_beta = (1-(prediction_weighted_sum/weight_sum)) * (reports)
            arm.influence_reward_dist.set_params(alpha_tilde + pre_alpha, beta_tilde + pre_beta)

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        return self.bandit.select_arm(t, influence_limit = influence_limit)
        #we should also use quantile for the predictions!

    def _update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            self.agent_loss[agent] += self.scoring_rule(reward, self.agency.agent_reports[agent][arm])

    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self._update_reputations(arm, reward)
        self._compute_T_posterior(arm, reward)
    
    def plot_reputations(self):
        for (agent, reputations) in self.agent_reputations_track.items():
            plt.plot(reputations, label=agent.id)
        plt.legend()
        plt.xlabel("Round (t)")
        plt.ylabel("Reputation")
        plt.show()

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2

