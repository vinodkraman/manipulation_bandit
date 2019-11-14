from BernoulliDistribution import BernoulliDistribution
from Agent import Agent

test = [0.1, 0.2, 0.3]
arm_dists = [BernoulliDistribution(i) for i in test]
agent = Agent(True, 0.1, arm_dists, 10000)

assert agent.reputation == 0.1
assert agent.trustworthy == True
assert agent.arm_dists == arm_dists
assert agent.num_reports == 10000

reports = agent.generate_reports()

assert len(reports) == 3
for index, report in enumerate(reports):
    assert(test[index] - 0.03 <= report <= test[index] + 0.03)

