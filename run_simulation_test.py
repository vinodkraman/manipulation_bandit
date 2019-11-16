from InfluenceLimiter import InfluenceLimiter
from NonInfluenceLimiter import NonInfluenceLimiter
from BayesianGreedy import BayesianGreedy
from BetaDistribution import BetaDistribution
from EpsilonGreedy import EpsilonGreedy
from BayesianUCB import BayesianUCB
from Nature import Nature
import matplotlib.pyplot as plt

T = 500
K = 5
num_agents = 5
num_reports = 10
initial_reputations = 1

nature = Nature(num_agents, K)
mal = [True, True, False, True, False]

theta = [0.8, 0.1, 0.1, 0.1, 0.1]

nature.initialize_arms()
nature.initialize_agents(mal, num_reports, initial_reputations)
# nature.agency.agent_reports = [[0.1, 0.3, 0.8], [0.9, 0.7, 0.2]]

bayesian_greedy_il = BayesianGreedy(K, epsilon=0.8)
bayesian_greedy_nonil = BayesianGreedy(K, epsilon=0.8)
influence_limiter = InfluenceLimiter(bayesian_greedy_il, nature.agency, num_reports)
non_influence_limiter = NonInfluenceLimiter(bayesian_greedy_nonil, nature.agency, num_reports)

epsilon = EpsilonGreedy(K, epsilon=0.8)
bayesian_ucb = BayesianUCB(K)

for i in range(K):
    reward = nature.generate_reward(i)
    epsilon.update_arm(i, reward)

total_reward = 0
total_epsilon_reward = 0
total_nonil_reward = 0
history = []
epsilon_history = []
nonil_history = []

arm_history = []
for t in range(T):
    reports = nature.get_agent_reports()
    # print(reports)

    influence_limiter.compute_IL_posterior()
    selected_arm = influence_limiter.select_arm(t)
    arm_history.append(selected_arm)
    reward = nature.generate_reward(selected_arm)
    total_reward += reward
    history.append(total_reward/(t+1))
    influence_limiter.update_reputations(selected_arm, reward)
    influence_limiter.compute_T_posterior(selected_arm, reward)

    non_influence_limiter.compute_IL_posterior()
    selected_arm = non_influence_limiter.select_arm(t)
    reward = nature.generate_reward(selected_arm)
    total_nonil_reward += reward
    nonil_history.append(total_nonil_reward/(t+1))
    non_influence_limiter.compute_T_posterior(selected_arm, reward)

    epsilon_arm = epsilon.select_arm(t+1)
    epsilon_reward = nature.generate_reward(epsilon_arm)
    epsilon.update_arm(epsilon_arm, epsilon_reward)
    total_epsilon_reward += epsilon_reward
    epsilon_history.append(total_epsilon_reward/(t+1))

[print(agent.reputation) for agent in nature.agency.agents]

plt.plot(history, label="influence-limit")
plt.plot(epsilon_history, label="epsilon greedy")
plt.plot(nonil_history, label="non_il")
plt.legend()
plt.show()



