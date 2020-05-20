# from bandits.bandit import Bandit
# from distributions.betadistribution import BetaDistribution
# import numpy as np
# import copy
# from scipy.stats import beta
# from scipy.special import betainc
# from scipy.special import beta
# import matplotlib.pyplot as plt
# from scipy.spatial import distance


# class Influencelimiter_freq_test():
#     def __init__(self, bandit, agency, reward_reports, initial_reputation, C, track_reputation= True):
#         self.bandit = bandit
#         self.agency = agency
#         self.posterior_history = {}
#         self.prediction_history = {}
#         self.reward_reports = reward_reports
#         self.initial_reputation = initial_reputation
#         self.track_reputation = track_reputation
#         self.q_tilde = []
#         self.C = C
#         super().__init__()
    
#     def reset(self):
#         self.bandit.reset()
#         self.posterior_history = {}
#         self.prediction_history = {}
#         self.__initialize_reputations()

#     def __initialize_reputations(self):
#         self.agent_reputations = {agent:self.initial_reputation for agent in self.agency.agents}
#         # self.agent_reputations = [int(agent.trustworthy == True) for agent in self.agency.agents]
#         if self.track_reputation:
#             self.agent_reputations_track = {agent:[self.initial_reputation] for agent in self.agency.agents}

#     def plot_posterior_history(self, arm):
#         x = np.linspace(0, 1.0, 100)
#         for (index, dist) in enumerate(self.prediction_history[arm]):
#             a, b = dist.get_params()
#             y = beta.pdf(x, a, b)
#             plt.plot(x, y, label=index)
#         plt.legend()
#         plt.show()

#     def _compute_SMA(self, arm_index):
#         npa = np.asarray(copy.deepcopy(self.agency.agent_reports))
#         return np.mean(npa[:,arm_index])
    
#     def _compute_IL_posterior(self, t):
#         self.q_tilde = []
#         for (arm_index, arm) in enumerate(self.bandit.arms):
#             self.prediction_history[arm_index]=[]

#             q_j_tilde = 0
#             if t <= self.bandit.K:
#                 q_j_tilde = 0.5
#             else:
#                 q_j_tilde = arm.mean_reward()

#             self.posterior_history[arm_index] = [copy.deepcopy(q_j_tilde)]

#             #iterate through each agent and process their report
#             for agent_index, agent in enumerate(self.agency.agents):
#                 # print(agent.id)
#                 gamma = min(self.agent_reputations[agent], 1)

#                 q_j = self.agency.agent_reports[agent][arm_index]
#                 self.prediction_history[arm_index].append(q_j)


#                 q_j_tilde = (1-gamma) * q_j_tilde + gamma * q_j
#                 self.posterior_history[arm_index].append(q_j_tilde)
    
#             self.q_tilde.append(q_j_tilde)


#     def select_arm(self, t, influence_limit = True):
#         self._compute_IL_posterior(t)
#         # print("predictions", self.q_tilde)
#         W = 0
#         max_rep = np.max([value for (agent, value) in self.agent_reputations.items()])
        
#         for agent, reputation in self.agent_reputations.items():
#             W += min(1,reputation)
        
#         test_arm = np.argmax(self.q_tilde)
#         return self.bandit.select_arm(t, self.q_tilde, W, self.C), 0
#         # return test_arm, 0
#         #we should also use quantile for the predictions!

#     def _update_reputations(self, arm, reward):
#         # [print(dist.mean()) for dist in self.posterior_history[arm]]
#         # eta = np.sqrt((8 * np.log(len(self.agency.agents)))/self.bandit.T)
#         for index, agent in enumerate(self.agency.agents):
#             gamma = min(1, self.agent_reputations[agent])
#             w = self.agent_reputations[agent]
#             q_tile_j_1 = self.posterior_history[arm][index]
#             q_j = self.prediction_history[arm][index]

#             # self.agent_reputations[agent] = w*np.exp(eta * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j)))

#             self.agent_reputations[agent] += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
#             if self.track_reputation == True:
#                 self.agent_reputations_track[agent].append(self.agent_reputations[agent])

#     def _compute_T_posterior(self, selected_arm, reward):
#         self.bandit.update(selected_arm, reward)

#     def update(self, arm, reward):
#         self._update_reputations(arm, reward)
#         self._compute_T_posterior(arm, reward)
    
#     def plot_reputations(self):
#         for (agent, reputations) in self.agent_reputations_track.items():
#             plt.plot(reputations, label=agent.id)
#         plt.legend()
#         # plt.ylim([0, 1])
#         plt.xlabel("Round (t)")
#         plt.ylabel("Reputation")
#         plt.show()

#     def scoring_rule(self, r, q, rule = "quadratic"):
#         return (r - q)**2
#         # if r == 1:
#         #     return (1-q)**2
#         # else:
#         #     return (q)**2
from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Influencelimiter_freq_test():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, C, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        self.q_tilde = []
        self.C = C
        self.goog = False
        self.random_order = np.arange(self.bandit.T)
        super().__init__()
    
    def reset(self):
        self.bandit.reset()
        self.posterior_history = {}
        self.prediction_history = {}
        self.__initialize_reputations()
        np.random.shuffle(self.random_order)

    def __initialize_reputations(self):
        # self.agent_reputations = {agent:self.initial_reputation for agent in self.agency.agents}
        self.agent_reputations_overall = {agent:self.initial_reputation for agent in self.agency.agents}
        self.agent_reputations = {agent: {arm:self.initial_reputation for arm in self.bandit.arms} for agent in self.agency.agents}
        

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

    def _compute_SMA(self, arm_index):
        npa = np.asarray(copy.deepcopy(self.agency.agent_reports))
        return np.mean(npa[:,arm_index])
    
    def _compute_IL_posterior(self, t):
        self.q_tilde = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            self.prediction_history[arm_index]=[]
            initial = 0.5
            if t <= self.bandit.K:
                initial = 0.5
            else:
                initial = 0.5
            
            self.posterior_history[arm_index] = [initial]

            weight_0 = 1
            weight = copy.deepcopy(weight_0) #have to make dependant on initial reputation and 
            running_sum = initial * weight
            true_weight = 0
            true_running_sum = 0

            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                # print(agent.id)
                gamma = min(self.agent_reputations[agent][arm], 1)

                temp_running_sum = running_sum + (self.agency.agent_reports[agent][arm_index])
                temp_weight = weight + 1

                q_j = temp_running_sum / temp_weight
                self.prediction_history[arm_index].append(q_j)

                running_sum += self.agency.agent_reports[agent][arm_index] * gamma
                weight += gamma

                true_running_sum += self.agency.agent_reports[agent][arm_index] * self.agent_reputations[agent][arm]
                true_weight += self.agent_reputations[agent][arm]

                q_j_tilde = running_sum/weight
                self.posterior_history[arm_index].append(q_j_tilde)
    
            running_sum -= 0.5 * weight_0
            weight -= weight_0
            # self.q_tilde.append(true_running_sum/true_weight)
            self.q_tilde.append(running_sum/weight)

        # print(self.q_tilde)
        # print(test)
        # print("distance", distance.euclidean(self.q_tilde, test))

    # def _compute_IL_posterior(self, t):
    #     self.q_tilde = []
    #     for (arm_index, arm) in enumerate(self.bandit.arms):
    #         self.prediction_history[arm_index]=[]

    #         q_j_tilde = 0
    #         if t <= self.bandit.K:
    #             q_j_tilde = 0.5
    #         else:
    #             q_j_tilde = arm.mean_reward()

    #         self.posterior_history[arm_index] = [copy.deepcopy(q_j_tilde)]

    #         #iterate through each agent and process their report
    #         for agent_index, agent in enumerate(self.agency.agents):
    #             # print(agent.id)
    #             gamma = min(self.agent_reputations[agent], 1)

    #             q_j = self.agency.agent_reports[agent][arm_index]
    #             self.prediction_history[arm_index].append(q_j)


    #             q_j_tilde = (1-gamma) * q_j_tilde + gamma * q_j
    #             self.posterior_history[arm_index].append(q_j_tilde)
    
    #         self.q_tilde.append(q_j_tilde)

    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)
        # print("predictions", self.q_tilde)
        W = 0
        # max_rep = np.max([value for (agent, value) in self.agent_reputations.items()])
        
        # for agent, reputation in self.agent_reputations.items():
        #     W += int(min(1,reputation) == 1)
        
        # test_arm = np.argmax(self.q_tilde)
        W = self.compute_overall_reputation_per_arm(self.agent_reputations)
        # print(W)
        selected_arm, prediction = self.bandit.select_arm(t, self.q_tilde, W, self.C, self.random_order)
        self.goog = prediction
        return selected_arm, 0
        # return test_arm, 0
        #we should also use quantile for the predictions!

    def _update_reputations(self, arm, reward):
        # [print(dist.mean()) for dist in self.posterior_history[arm]]
        # eta = np.sqrt((8 * np.log(len(self.agency.agents)))/self.bandit.T)
        for index, agent in enumerate(self.agency.agents):
            gamma = min(1, self.agent_reputations[agent][self.bandit.arms[arm]])
            w = self.agent_reputations[agent]
            q_tile_j_1 = self.posterior_history[arm][index]
            q_j = self.prediction_history[arm][index]

            # self.agent_reputations[agent] = w*np.exp(eta * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j)))
            gamma_2 = min(1, self.agent_reputations_overall[agent])
            self.agent_reputations_overall[agent] += gamma_2 * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
            self.agent_reputations[agent][self.bandit.arms[arm]] += gamma * (self.scoring_rule(reward, q_tile_j_1) - self.scoring_rule(reward, q_j))
            if self.track_reputation == True:
                self.agent_reputations_track[agent].append(self.agent_reputations_overall[agent])
        # print(self.compute_overall_reputation(self.agent_reputations))
        
    def compute_overall_reputation(self, agent_reputations):
        reps = []
        for agent, reputations in agent_reputations.items():
            rep_sum = 0
            for arm, rep in reputations.items():
                rep_sum += rep
            reps.append((agent.id, rep_sum))

        return reps

    def compute_overall_reputation_per_arm(self, agent_reputations):
        reps = {arm:0 for arm in self.bandit.arms}
        for agent, reputations in agent_reputations.items():
            for arm, rep in reputations.items():
                reps[arm] += min(rep, 1)

        return reps



    def _compute_T_posterior(self, selected_arm, reward):
        self.bandit.update(selected_arm, reward)

    def update(self, arm, reward):
        self._update_reputations(arm, reward)
        # if self.goog != 2:
        # print(self.goog)
        # print(arm)
        
        # if self.goog == arm:
        # W = self.compute_overall_reputation_per_arm(self.agent_reputations)
        W = 0
        # max_rep = np.max([value for (agent, value) in self.agent_reputations.items()])
        
        for agent, reputation in self.agent_reputations_overall.items():
            W += int(min(1,reputation) == 1)

        if W > 1:
            self.bandit.arms[self.goog].real_pulls += 1

        self._compute_T_posterior(arm, reward)

        # if self.goog <= 0:
        #     self.bandit.arms[arm].real_pulls += reward
        # # print("STATUS:", self.goog)
        # # should_explore = np.random.rand()
        # # if should_explore < self.goog or self.goog == -1:
        # # if self.goog == False:
        #     # self._update_reputations(arm, reward)
        # if self.goog >= 0:
        #     self._compute_T_posterior(arm, reward)
    
    def plot_reputations(self):
        for (agent, reputations) in self.agent_reputations_track.items():
            plt.plot(reputations, label=agent.id)
        plt.legend()
        # plt.ylim([0, 1])
        plt.xlabel("Round (t)")
        plt.ylabel("Reputation")
        plt.show()

    def scoring_rule(self, r, q, rule = "quadratic"):
        return (r - q)**2
