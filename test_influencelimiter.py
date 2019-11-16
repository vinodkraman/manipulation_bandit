from InfluenceLimiter import InfluenceLimiter
from BayesianGreedy import BayesianGreedy
from BetaDistribution import BetaDistribution
from Nature import Nature

K = 3
num_agents = 3
num_reports = 10
initial_reputations = 0.1

nature = Nature(num_agents, K)
theta = [0.1, 0.2, 0.8]
mal = [False, False, False]

nature.initialize_arms(theta)
nature.initialize_agents(mal, num_reports, initial_reputations)

print(nature.agency.agents[0].arm_dists[0].theta)
print(nature.agency.agents[0].arm_dists[1].theta)
print(nature.agency.agents[0].arm_dists[2].theta)

print(nature.agency.agents[1].arm_dists[0].theta)
print(nature.agency.agents[1].arm_dists[1].theta)
print(nature.agency.agents[1].arm_dists[2].theta)

print(nature.agency.agents[2].arm_dists[0].theta)
print(nature.agency.agents[2].arm_dists[1].theta)
print(nature.agency.agents[2].arm_dists[2].theta)

nature.agency.agent_reports = [[0.2, 0.1, 0.7], [0.1, 0.3, 0.9], [0.2, 0.2, 0.6]]

bayesian_greedy = BayesianGreedy(K, epsilon=1)

influence_limiter = InfluenceLimiter(bayesian_greedy, nature.agency, num_reports)
# gamma = min(1, agent.reputation)
# alpha_tilde = (1-gamma) * alpha_tilde + gamma*self.agency.agent_reports[index][arm_index]*agent.num_reports
# beta_tilde = (1-gamma) * beta_tilde + gamma*(1-self.agency.agent_reports[index][arm_index])*agent.num_reports

influence_limiter.compute_IL_posterior()
arm0_influence_alpha_0 =  (1-0.1)*1 + 0.1 * 0.1 * 2 
arm0_influence_beta_0 =  (1-0.1)*1 + 0.1 * (1-0.1) * 2 
arm0_influence_alpha_1 = (1-0.1) * arm0_influence_alpha_0 + 0.1 * 0.9 * 2
arm0_influence_beta_1 = (1-0.1) * arm0_influence_beta_0 + 0.1 * (1-0.9) * 2

arm1_influence_alpha_0 =  (1-0.1)*1 + 0.1 * 0.3 * 2 
arm1_influence_beta_0 =  (1-0.1)*1 + 0.1 * (1-0.3) * 2 
arm1_influence_alpha_1 = (1-0.1) * arm1_influence_alpha_0 + 0.1 * 0.7 * 2
arm1_influence_beta_1 = (1-0.1) * arm1_influence_beta_0 + 0.1 * (1-0.7) * 2

arm2_influence_alpha_0 =  (1-0.1)*1 + 0.1 * 0.8 * 2 
arm2_influence_beta_0 =  (1-0.1)*1 + 0.1 * (1-0.8) * 2 
arm2_influence_alpha_1 = (1-0.1) * arm2_influence_alpha_0 + 0.1 * 0.2 * 2
arm2_influence_beta_1 = (1-0.1) * arm2_influence_beta_0 + 0.1 * (1-0.2) * 2

print(influence_limiter.bandit.arms[0].influence_reward_dist.alpha)
print(influence_limiter.bandit.arms[0].influence_reward_dist.beta)
for item in influence_limiter.posterior_history[0]:
    print(item.get_params())

print(influence_limiter.bandit.arms[1].influence_reward_dist.alpha)
print(influence_limiter.bandit.arms[1].influence_reward_dist.beta)
for item in influence_limiter.posterior_history[1]:
    print(item.get_params())

print(influence_limiter.bandit.arms[2].influence_reward_dist.alpha)
print(influence_limiter.bandit.arms[2].influence_reward_dist.beta)
for item in influence_limiter.posterior_history[2]:
    print(item.get_params())


# assert(influence_limiter.bandit.arms[0].influence_reward_dist.alpha == arm0_influence_alpha_1)
# assert(influence_limiter.bandit.arms[0].influence_reward_dist.beta == arm0_influence_beta_1)
# assert(influence_limiter.bandit.arms[1].influence_reward_dist.alpha == arm1_influence_alpha_1)
# assert(influence_limiter.bandit.arms[1].influence_reward_dist.beta == arm1_influence_beta_1)
# assert(influence_limiter.bandit.arms[2].influence_reward_dist.alpha == arm2_influence_alpha_1)
# assert(influence_limiter.bandit.arms[2].influence_reward_dist.beta == arm2_influence_beta_1)

selected_arm = influence_limiter.select_arm()
print(selected_arm)
assert(selected_arm == 2)

reward = 1
influence_limiter.update_reputations(selected_arm, reward)

print(nature.agency.agents[0].reputation)
print(nature.agency.agents[1].reputation)
print(nature.agency.agents[2].reputation)

influence_limiter.compute_T_posterior(selected_arm, reward)
print(influence_limiter.bandit.arms[selected_arm].reward_dist.get_params())


