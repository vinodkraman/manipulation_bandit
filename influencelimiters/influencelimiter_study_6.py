from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt

class InfluenceLimiter_study_6():
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
        for (arm_index, arm) in enumerate(self.bandit.arms):
            self.prediction_history[arm_index]=[]
            alpha_tilde = 1
            beta_tilde = 1
            self.posterior_history[arm_index] = [BetaDistribution(copy.deepcopy(alpha_tilde), copy.deepcopy(beta_tilde))]
            pre_alpha, pre_beta = copy.deepcopy(arm.reward_dist.get_params())

            running_alpha_sum = 0
            running_beta_sum = 0
            weight = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                gamma = min(1, self.agent_reputations[agent_index]) #get gamma
                

                #give full weight to currnt agents reports
                temp_weight = weight + 1
                temp_running_alpha_sum = running_alpha_sum + (self.agency.agent_reports[agent_index][arm_index] * agent.num_reports)
                temp_running_beta_sum = running_beta_sum + (1-self.agency.agent_reports[agent_index][arm_index]) * agent.num_reports
                alpha_j = temp_running_alpha_sum / temp_weight
                beta_j = temp_running_beta_sum / temp_weight

                #update weighted sum
                running_alpha_sum += self.agency.agent_reports[agent_index][arm_index] * gamma * agent.num_reports
                running_alpha_sum += (1-self.agency.agent_reports[agent_index][arm_index]) * gamma * agent.num_reports
                weight += gamma

                self.prediction_history[arm_index].append(BetaDistribution(copy.deepcopy(alpha_j), copy.deepcopy(beta_j)).get_quantile(0.95))


                alpha_tilde = (1-gamma) * alpha_tilde + gamma * alpha_j
                beta_tilde = (1-gamma) * beta_tilde + gamma * beta_j

                # print("params", alpha_tilde, beta_tilde)

                self.posterior_history[arm_index].append(BetaDistribution(copy.deepcopy(alpha_tilde), copy.deepcopy(beta_tilde).get_quantile(0.95)))
    
            arm.influence_reward_dist.set_params(alpha_tilde + pre_alpha, beta_tilde + pre_beta)


    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        return self.bandit.select_arm(t, influence_limit = influence_limit)
        #we should also use quantile for the predictions!

    def _update_reputations(self, arm, reward):
        # [print(dist.mean()) for dist in self.posterior_history[arm]]
        for index, agent in enumerate(self.agency.agents):
            
            gamma = min(1, self.agent_reputations[index])
            q_tile_j_1 = self.posterior_history[arm][index]
            q_j = self.prediction_history[arm][index]

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

