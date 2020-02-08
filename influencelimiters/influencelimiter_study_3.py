from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
import matplotlib.pyplot as plt

class InfluenceLimiter_study_3():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.prediction_history = {}
        self.__initialize_reputations()

    def __initialize_reputations(self):
        self.agent_reputations = [self.initial_reputation for agent in self.agency.agents]
        # self.agent_reputations[0] = 0.5
        # self.agent_reputations = [int(agent.trustworthy == True) for agent in self.agency.agents]
        if self.track_reputation:
            self.agent_reputations_track = [[self.initial_reputation] for agent in self.agency.agents]

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
        # print("reputations:", self.agent_reputations)
        for (arm_index, arm) in enumerate(self.bandit.arms):
            # self.posterior_history[arm_index] = [BetaDistribution(1, 1)]
            self.prediction_history[arm_index]=[]

            pre_alpha, pre_beta = copy.deepcopy(arm.reward_dist.get_params())
            # self.posterior_history[arm_index] = [copy.deepcopy(arm.reward_dist)]
            # k = 2/(len(self.agency.agents) + 1)
            pre_mean = copy.deepcopy(arm.reward_dist.mean())
            prev_ema = copy.deepcopy(self.agency.agent_reports[0][arm_index])
            # q_j_tilde = copy.deepcopy(0.5)
            k = 0.75 
            self.posterior_history[arm_index] = [BetaDistribution(0.5, 1 - 0.5)]
            q_j_tilde = copy.deepcopy(0.5)

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                # print("agent:", agent_index)
                # print("agent reputation:", self.agent_reputations[agent_index])
                k = 1 - 1/(t + 1)
                gamma = min(1, self.agent_reputations[agent_index])
                current_ema = (self.agency.agent_reports[agent_index][arm_index] - prev_ema) * k + prev_ema
                prev_ema = copy.deepcopy(current_ema)

                alpha_j = current_ema * (agent.num_reports) #+ pre_alpha
                beta_j = (1-current_ema) * (agent.num_reports) # pre_beta


                self.prediction_history[arm_index].append(BetaDistribution(alpha_j, beta_j))

                q_j = copy.deepcopy(alpha_j/(alpha_j + beta_j))
                q_j_tilde = (1-gamma)*q_j_tilde + gamma*(q_j)

                alpha_tilde = q_j_tilde * (agent.num_reports) 
                beta_tilde = (1-q_j_tilde) * (agent.num_reports)
                self.posterior_history[arm_index].append(BetaDistribution(alpha_tilde, beta_tilde))
    
            # print("final:", alpha_tilde + pre_alpha, beta_tilde + pre_beta)
            arm.influence_reward_dist.set_params(alpha_tilde + pre_alpha, beta_tilde + pre_beta)

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        return self.bandit.select_arm(t, influence_limit = influence_limit)

    def _update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            gamma = min(1, self.agent_reputations[index])
            q_tile_j_1 = self.posterior_history[arm][index].mean()
            q_j = self.prediction_history[arm][index].mean()
            
            self.agent_reputations[index] += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
            if self.track_reputation == True:
                self.agent_reputations_track[index].append(self.agent_reputations[index])

    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        # print("pre_rep update:", self.agent_reputations)
        self._update_reputations(arm, reward)
        # print("post_rep update:", self.agent_reputations)
        self._compute_T_posterior(arm, reward)
    
    def plot_reputations(self):
        for (index, reputations) in enumerate(self.agent_reputations_track):
            plt.plot(reputations, label=index)
        plt.legend()
        plt.xlabel("Round (t)")
        plt.ylabel("Reputation")
        plt.show()

    def scoring_rule(self, r, q, rule = "quadratic"):
        if r == 1:
            return (1-q)**2
        else:
            return (q)**2

