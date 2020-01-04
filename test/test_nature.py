from Nature import Nature
from BayesianGreedy import BayesianGreedy
from InfluenceLimiter import InfluenceLimiter


T = 2
K = 5
num_agents = 5
num_reports = 5
initial_reputations = 0.1

nature = Nature(num_agents, K)
theta = [0.1, 0.2, 0.8, 0.2, 0.3]
mal = [True, True, False, False, False]

#TEST: initialize_arms
nature.initialize_arms(theta)
assert nature.hidden_theta == theta
assert nature.best_arm_mean == 0.8
for index, dist in enumerate(nature.arm_dists):
    assert dist.theta == theta[index]
for index, dist in enumerate(nature.malicious_dists):
    assert dist.theta == 1 - theta[index]
assert len(nature.hidden_theta) == K

#TEST: initialize agents
nature.initialize_agents(mal, num_reports, initial_reputations)
assert len(nature.agency.agents) == num_agents
for index, agent in enumerate(nature.agency.agents):
    assert agent.reputation == initial_reputations
    if mal[index] == True:
        assert agent.arm_dists == nature.malicious_dists
    else:
        assert agent.arm_dists == nature.arm_dists
    
    assert agent.num_reports == num_reports

#TEST: get_agent_reports
reports = nature.get_agent_reports()
assert len(reports) == num_agents

#Test: get rewards
trials = 10000
for index, arm in enumerate(theta):
    total = 0
    for i in range(trials):
        total += nature.generate_reward(index)

    assert(theta[index] - 0.03 <= total/trials <= theta[index] + 0.03)

# bayesian_greedy = BayesianGreedy(K, epsilon=0.7)
# il = InfluenceLimiter(bayesian_greedy, nature.agency, num_reports)


# nature.get_agent_reports()
# il.compute_IL_posterior()
# arm = il.select_arm()
# reward = nature.generate_reward(arm)
# il.update_reputations(arm, reward)
# il.compute_T_posterior(arm, reward)



