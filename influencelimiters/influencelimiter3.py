from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
import matplotlib.pyplot as plt

class InfluenceLimiter3():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.__initialize_reputations()

    def __initialize_reputations(self):
        self.agent_reputations = [self.initial_reputation for agent in self.agency.agents]
        if self.track_reputation:
            self.agent_reputations_track = [[self.initial_reputation] for agent in self.agency.agents]
        # for agent in self.agency.agents:
        #     agent.reputation = self.initial_reputation

    def plot_posterior_history(self, arm):
        x = np.linspace(0, 1.0, 100)
        for (index, dist) in enumerate(self.posterior_history[arm]):
            a, b = dist.get_params()
            y = beta.pdf(x, a, b)
            plt.plot(x, y, label=index)
        plt.legend()
        plt.show()
        
    def _compute_IL_posterior(self):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            self.posterior_history[arm_index] = [copy.deepcopy(arm.reward_dist)]

            alpha_tilde, beta_tilde = arm.reward_dist.get_params()

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                # print(agent.reputation)
                # gamma = min(1, agent.reputation)
                gamma = min(1, self.agent_reputations[agent_index])
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[agent_index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[agent_index][arm_index])*agent.num_reports
                self.posterior_history[arm_index].append(BetaDistribution(alpha_tilde, beta_tilde))

            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)
            #compute posterior and set to bandit influence-limited posterior
    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior()
        return self.bandit.select_arm(t, influence_limit = influence_limit)

    def _update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            # gamma = min(1, agent.reputation)
            gamma = min(1, self.agent_reputations[index])
            q_tile_j_1 = self.posterior_history[arm][index].mean()
            alpha_delta = 0
            beta_delta = 0
            for i in range(index + 1):
                alpha_delta += self.agency.agent_reports[i][arm] * agent.num_reports 
                beta_delta += (1-self.agency.agent_reports[i][arm]) * agent.num_reports 

            prev_alpha, prev_beta = self.bandit.arms[arm].reward_dist.get_params()
            q_j = BetaDistribution(prev_alpha + alpha_delta, prev_beta + beta_delta).mean()
            # prev_alpha, prev_beta = self.posterior_history[arm][0].get_params()
            # q_j = (prev_alpha + alpha_delta)/ (prev_alpha + prev_beta + agent.num_reports * (index + 1))

            # agent.reputation += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
            self.agent_reputations[index] += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
            if self.track_reputation == True:
                self.agent_reputations_track[index].append(self.agent_reputations[index])

    # def _compute_T_posterior(self, selected_arm, reward):
    #     # self.bandit.arms[selected_arm].reward_dist.update(reward)
    #     for (arm_index, arm) in enumerate(self.bandit.arms):
    #         alpha_tilde, beta_tilde = arm.reward_dist.get_params()
    #         # alpha_tilde = 0
    #         # beta_tilde = 0
    #         for index, agent in enumerate(self.agency.agents):
    #             gamma = min(1, agent.reputation)
    #             alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
    #             beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports

    #         if arm_index == selected_arm:
    #             alpha_tilde += (reward == 1) * self.reward_reports
    #             beta_tilde += (reward == 0) * self.reward_reports
    #         # self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
    #         arm.reward_dist.set_params(alpha_tilde, beta_tilde)
    #     # self.bandit.arms[selected_arm].reward_dist.update(reward)

    def _compute_T_posterior(self, selected_arm, reward):
        for (arm_index, arm) in enumerate(self.bandit.arms):
            alpha_tilde, beta_tilde = arm.reward_dist.get_params()
            # alpha_tilde = 0
            # beta_tilde = 0
            for index, agent in enumerate(self.agency.agents):
                # gamma = min(1, agent.reputation)
                gamma = min(1, self.agent_reputations[index])
                alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
                beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports

            if arm_index == selected_arm:
                alpha_tilde += (reward == 1) * self.reward_reports
                beta_tilde += (reward == 0) * self.reward_reports
            # self.bandit.arms[arm].reward_dist.update(alpha_tilde + reward_alpha, beta_tilde + reward_beta)
            arm.reward_dist.set_params(alpha_tilde, beta_tilde)
        # self.bandit.arms[selected_arm].reward_dist.update(reward)

    def update(self, arm, reward):
        self._update_reputations(arm, reward)
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