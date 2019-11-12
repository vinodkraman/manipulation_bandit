from Agent import Agent
class Agency():
    def __init__(self):
        self.agents = []
        self.agent_reports = []

    def create_agent(self, trust, arm_dists, num_reports, initial_reputation):
        self.agents.append(Agent(trust, initial_reputation, arm_dists, num_reports))

    def send_reports(self):
        reports = []
        for agent in self.agents:
            reports.append(agent.generate_reports())

        self.agent_reports = reports 
        return reports
        

#truthful to one arm, malicious to another?