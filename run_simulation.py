from Nature import Nature
from BayesianGreedy import BayesianGreedy
from InfluenceLimiter import InfluenceLimiter


T = 2
K = 5
num_agents = 5
num_reports = 5
initial_reputations = 0.1

nature = Nature(num_agents, K)
nature.initialize_arms()
mal = [True, True, False, False, False]

nature.initialize_agents(mal, num_reports, initial_reputations)
reports = nature.get_agent_reports()
bayesian_greedy = BayesianGreedy(K, epsilon=0.7)
il = InfluenceLimiter(bayesian_greedy, nature.agency, num_reports)


nature.get_agent_reports()
il.compute_IL_posterior()
arm = il.select_arm()
reward = nature.generate_reward(arm)
il.update_reputations(arm, reward)
il.compute_T_posterior(arm, reward)



