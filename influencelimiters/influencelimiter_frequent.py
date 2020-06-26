from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
from scipy.special import beta
import matplotlib.pyplot as plt
from scipy.spatial import distance


class Influencelimiter_frequent():
    def __init__(self, bandit, agency, reward_reports, initial_reputation, C, track_reputation= True):
        self.bandit = bandit
        self.agency = agency
        self.posterior_history = {}
        self.prediction_history = {}
        self.reward_reports = reward_reports
        self.initial_reputation = initial_reputation
        self.track_reputation = track_reputation
        self.q_tilde = []
        self.reports = []
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
        self.agent_reputations_overall = {agent:self.initial_reputation for agent in self.agency.agents}
        self.agent_reputations = {agent: {arm:self.initial_reputation for arm in self.bandit.arms} for agent in self.agency.agents}
        

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
    
    def _compute_IL_posterior(self, t):
        self.q_tilde = []
        self.reports = []
        for (arm_index, arm) in enumerate(self.bandit.arms):
            self.prediction_history[arm_index]=[]
            initial = 0.5
            if t > self.bandit.K:
                initial = 0.5         
            self.posterior_history[arm_index] = [initial]

            weight_0 = 1
            weight = copy.deepcopy(weight_0) #have to make dependant on initial reputation and 
            running_sum = initial * weight
            num_reports = 0


            #iterate through each agent and process their report
            for agent_index, agent in enumerate(self.agency.agents):
                # print(agent.id)
                gamma = min(self.agent_reputations[agent][arm], 1)
                num_reports += gamma * agent.num_reports

                temp_running_sum = running_sum + (self.agency.agent_reports[agent][arm_index])
                temp_weight = weight + 1

                q_j = temp_running_sum / temp_weight
                self.prediction_history[arm_index].append(q_j)

                running_sum += self.agency.agent_reports[agent][arm_index] * gamma
                weight += gamma

                q_j_tilde = running_sum/weight
                self.posterior_history[arm_index].append(q_j_tilde)
    
            running_sum -= 0.5 * weight_0
            weight -= weight_0

            self.q_tilde.append(running_sum/weight)
            self.reports.append(num_reports)


    def select_arm(self, t, influence_limit = True):
        self._compute_IL_posterior(t)

        # W = self.compute_overall_reputation_per_arm(self.agent_reputations)
        W = 0
        for agent, reputation in self.agent_reputations_overall.items():
            W += int(min(1, reputation) == 1)
        selected_arm = self.bandit.select_arm(t, self.q_tilde, self.reports, W)
        return selected_arm, 0

    def _update_reputations(self, arm, reward):
        for index, agent in enumerate(self.agency.agents):
            gamma = min(1, self.agent_reputations[agent][self.bandit.arms[arm]])
            w = self.agent_reputations[agent]
            q_tile_j_1 = self.posterior_history[arm][index]
            q_j = self.prediction_history[arm][index]

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
        self._compute_T_posterior(arm, reward)
    
    def plot_reputations(self):
        for (agent, reputations) in self.agent_reputations_track.items():
            plt.plot(reputations, label=agent.id)
        plt.legend()
        plt.xlabel("Round (t)")
        plt.ylabel("Reputation")
        plt.show()

    def scoring_rule(self, r, q, rule = "quadratic"):
        return (r - q)**2
