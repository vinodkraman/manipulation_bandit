from InfluenceLimiter import InfluenceLimiter
from NonInfluenceLimiter import NonInfluenceLimiter
from BetaDistribution import BetaDistribution
from BayesUCB import BayesUCB
from scipy.stats import bernoulli
from Random import Random
from Nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from ThompsonSampling import ThompsonSampling 

T = 500
K = 5
num_agents = 5
num_reports = 10
initial_reputations = 1
num_exp = 1

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, num_agents=num_agents)
trust = [True, True, True, True, True]

bayes_ucb_il = BayesUCB(T, K, world_priors)
influence_limiter = InfluenceLimiter(bayes_ucb_il, nature.agency, num_reports)

cumulative_reward_history = {exp:np.zeros(T) for exp in range(num_exp)}
total_reward = {exp:0 for exp in range(num_exp)}

for exp in range(num_exp):
    nature.initialize_arms()
    nature.initialize_agents(trust, num_reports, initial_reputations)
    influence_limiter.reset()
    for t in range(T):
        reports = nature.get_agent_reports()
        selected_arm = influence_limiter.select_arm(t+1)
        reward = nature.generate_reward(selected_arm)
        total_reward[exp] += reward
        cumulative_reward_history[exp][t] = total_reward[exp]/(t+1)
        influence_limiter.update_arm(selected_arm, reward)

[print(agent.reputation) for agent in nature.agency.agents]

plt.plot(cumulative_reward_history[0], label="influence-limit")
plt.legend()
plt.show()