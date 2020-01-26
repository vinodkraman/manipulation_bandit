from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
import matplotlib.pyplot as plt

class InfluenceLimiter2test15():
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
        
    def _compute_IL_posterior(self):
        # print("reputations:", self.agent_reputations)
        for (arm_index, arm) in enumerate(self.bandit.arms):
            self.posterior_history[arm_index] = [copy.deepcopy(arm.reward_dist)]
            self.prediction_history[arm_index]=[]

            # alpha_tilde, beta_tilde = copy.deepcopy(arm.reward_dist.get_params())
            # pre_mean = copy.deepcopy(arm.reward_dist.mean())
            pre_alpha, pre_beta = copy.deepcopy(arm.reward_dist.get_params())
        
            weight = 1
            running_weighted_sum = copy.deepcopy(arm.reward_dist.mean())
            # running_alpha_sum = copy.deepcopy(pre_alpha)
            # running_beta_sum = copy.deepcopy(pre_beta)
            # weights = 1
        
            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                gamma = min(1, self.agent_reputations[agent_index])

                alpha_j = self.agency.agent_reports[agent_index][arm_index] * (agent.num_reports + pre_alpha + pre_beta)
                beta_j = (1-self.agency.agent_reports[agent_index][arm_index]) * (agent.num_reports + pre_beta + pre_alpha)

                self.prediction_history[arm_index].append(BetaDistribution(alpha_j, beta_j))

                running_weighted_sum += gamma * self.agency.agent_reports[agent_index][arm_index]
                weight += gamma

                # running_alpha_sum += gamma * self.agency.agent_reports[agent_index][arm_index] * (agent.num_reports)
                # running_beta_sum += gamma * (1-self.agency.agent_reports[agent_index][arm_index]) * (agent.num_reports)
                # weights += gamma

                # alpha_tilde = running_alpha_sum/weights
                # beta_tilde = running_beta_sum/weights

                q_tilde = running_weighted_sum/weight
                alpha_tilde = q_tilde * (agent.num_reports + pre_alpha + pre_beta)
                beta_tilde = (1-q_tilde) * (agent.num_reports + pre_beta + pre_alpha)
                self.posterior_history[arm_index].append(BetaDistribution(alpha_tilde, beta_tilde))
    
            arm.influence_reward_dist.set_params(alpha_tilde, beta_tilde)

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior()
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


#ISSUE 1: Initializing Tilde
#initializing the prior to the reward is problematic for good users after a certain point (i.e. when arm pulls
# exceeps # of reports)
#initializing tilde to uninformative prior is problematic since bad users will gain reputation quickly if previous good agents 
#has low reputation
#We need a tilde that is not as smart as the current arm pull but not dumb enough as uniformative prior. 
#Computing posterior is problem

#ISSUE2: Posterior update
# If we update the posterior to large alpha and beta, then this slows growth of reputations for good users
#since diminishing returns

#ISSUE3: Computing q_j
#If we treat each agents advice as its own q_j (independently), then we need to aggregate over all good users.
#. This makes it hard if a good agent has reputation close to 1 but not above 1, we lose this agents advice completely, even though even partial taking the advice is useful!
# But how should we compute q_j that accounts for all agents? Sequentially is problematic since diminishing returns for agents near the end!
# 
# 
# neeed diminishing returns for weighted sum

#decouple magnitude of alpha and beta from 

