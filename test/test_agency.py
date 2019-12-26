from Agent import Agent
from Agency import Agency
from BernoulliDistribution import BernoulliDistribution 


#Test 
trust = True
test1 = [0.1, 0.2, 0.3]
test2 = [0.8, 0.8, 0.7]
arm_dists1 = [BernoulliDistribution(i) for i in test1]
arm_dists2 = [BernoulliDistribution(i) for i in test2]
num_reports = 10000
rep = 0.1

agency = Agency()
agency.create_agent(True, arm_dists1, num_reports, rep)
assert(len(agency.agents) == 1)
agency.create_agent(False, arm_dists2, num_reports, rep)
assert(len(agency.agents) == 2)

reports = agency.send_reports()
assert(len(reports) == len(agency.agents))
assert(len(reports[0]) == len(test1))
assert(len(reports[1]) == len(test2))

for index, report in enumerate(reports[0]):
    assert(test1[index] - 0.03 <= report <= test1[index] + 0.03)

for index, report in enumerate(reports[1]):
    assert(test2[index] - 0.03 <= report <= test2[index] + 0.03)







# def __init__(self):
#     self.agents = []
#     self.agent_reports = []

# def create_agent(self, trust, arm_dists, num_reports, initial_reputation):
#     self.agents.append(Agent(trust, initial_reputation, arm_dists, num_reports))

# def send_reports(self):
#     reports = []
#     for agent in self.agents:
#         reports.append(agent.generate_reports())

#     self.agent_reports = reports 
#     return reports
    