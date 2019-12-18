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

T = 200
K = 5
num_agents = 5
num_reports = 100
initial_reputations = 1

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, num_agents=num_agents)
trust = [True, True, False, True, False]

nature.initialize_arms()
nature.initialize_agents(trust, num_reports, initial_reputations)

bayes_ucb_il = BayesUCB(T, K, world_priors)
bayes_ucb_noil = BayesUCB(T, K, world_priors)
influence_limiter = InfluenceLimiter(bayes_ucb_il, nature.agency, num_reports)
non_influence_limiter = NonInfluenceLimiter(bayes_ucb_noil, nature.agency, num_reports)

total_reward = 0
total_nonil_reward = 0
history = []
nonil_history = []

arm_history = []
for t in range(T):
    reports = nature.get_agent_reports()

    influence_limiter.compute_IL_posterior()
    selected_arm = influence_limiter.select_arm(t+1)
    reward = nature.generate_reward(selected_arm)
    total_reward += reward
    history.append(total_reward/(t+1))
    influence_limiter.update_reputations(selected_arm, reward)
    influence_limiter.compute_T_posterior(selected_arm, reward)

    non_influence_limiter.compute_IL_posterior()
    selected_arm = non_influence_limiter.select_arm(t+1)
    reward = nature.generate_reward(selected_arm)
    total_nonil_reward += reward
    nonil_history.append(total_nonil_reward/(t+1))
    non_influence_limiter.compute_T_posterior(selected_arm, reward)

[print(agent.reputation) for agent in nature.agency.agents]

plt.plot(history, label="influence-limit")
plt.plot(nonil_history, label="non_il")
plt.legend()
plt.show()