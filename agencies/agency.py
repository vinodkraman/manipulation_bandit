from agencies.agent import Agent
import matplotlib.pyplot as plt
import numpy as np
import copy

class Agency():
    def __init__(self):
        self.agents = []
        self.agent_reports = []
        self.agent_reputations = []

    def create_agent(self, trust, arm_dists, num_reports, best_arm, target_arms, attack_freq, agent_id):
        self.agents.append(Agent(trust, arm_dists, num_reports, best_arm, target_arms, attack_freq, agent_id))
        self.agent_reputations.append([])

    def shuffle_agents(self):
        np.random.shuffle(self.agents)

    def send_reports(self, t, attack):
        reports = {}
        for index, agent in enumerate(self.agents):
            reports[agent] = agent.generate_reports_v2(t, attack, self.agents[:index], reports)

        # reports = [agent.generate_reports() for agent in self.agents]
        self.agent_reports = reports 
        return reports

    def clear_agents(self):
        self.agents = []
        self.agent_reports = []
        self.agent_reputations = []

    def track_reputations(self):
        for (index, agent) in enumerate(self.agents):
            self.agent_reputations[index].append(agent.reputation)

    def plot_reputations(self, agent=None, experiment=None):
        if agent != None:
            plt.plot(self.agent_reputations[agent], label=experiment)
            plt.legend()
            plt.xlabel("Round (t)")
            plt.ylabel("Reputation")
        else:
            for (index, reputations) in enumerate(self.agent_reputations):
                plt.plot(reputations, label=index)
            plt.legend()
            plt.xlabel("Round (t)")
            plt.ylabel("Reputation")
            plt.show()