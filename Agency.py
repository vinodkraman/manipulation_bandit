from Agent import Agent
import matplotlib.pyplot as plt
class Agency():
    def __init__(self):
        self.agents = []
        self.agent_reports = []
        self.agent_reputations = []

    def create_agent(self, trust, arm_dists, num_reports, initial_reputation):
        self.agents.append(Agent(trust, initial_reputation, arm_dists, num_reports))
        self.agent_reputations.append([])

    def send_reports(self):
        reports = []
        for agent in self.agents:
            reports.append(agent.generate_reports())

        self.agent_reports = reports 
        return reports

    def clear_agents(self):
        self.agents = []
        self.agent_reports = []
        self.agent_reputations = []

    def track_reputations(self):
        for (index, agent) in enumerate(self.agents):
            self.agent_reputations[index].append(agent.reputation)

    def plot_reputations(self):
        for (index, reputations) in enumerate(self.agent_reputations):
            plt.plot(reputations, label=index)

        plt.legend()
        plt.xlabel("Round (t)")
        plt.ylabel("Reputation")
        plt.show()