import numpy as np
from random import *
from scipy.stats import bernoulli
from UCB1 import UCB1
from Random import Random
from EpsilonGreedy import EpsilonGreedy
from BayesianGreedy import BayesianGreedy
from BayesianUCB import BayesianUCB
import matplotlib.pyplot as plt
from ThompsonSampling import ThompsonSampling 

#number of rounds
T = 10000
#number of arms
K = 5

hidden_dist = [random() for i in range(K)]
# hidden_dist = [0.1, 0.2, 0.1, 0.9, 0.1]
best_arm_mean = max(hidden_dist)

ucb = UCB1(K)
random = Random(K)
epsilon = EpsilonGreedy(K, epsilon=0.7)
thompson = ThompsonSampling(K)
bayesian_greedy = BayesianGreedy(K, epsilon=0.7)
bayesian_ucb = BayesianUCB(K)

bandits = [random, ucb, epsilon, thompson, bayesian_greedy, bayesian_ucb]

key_map = {ucb: "ucb", random: "random", epsilon:"epsilon", thompson: "thompson", bayesian_greedy: "bayesian_greedy", bayesian_ucb: "bayesian_ucb"}
# key_map = {ucb: "ucb", random: "random", epsilon:"epsilon", thompson: "thompson"}

cumulative_regret_history = {i:np.zeros(T) for i in bandits}
total_regret = {i:0 for i in bandits}

#pull each arm once
for i in range(K):
    reward = np.asscalar(bernoulli.rvs(hidden_dist[i], size=1))
    for bandit in bandits:
        bandit.update_arm(i, reward)

#run bandit
for t in range(T):
    for bandit in bandits:
        arm = bandit.select_arm(t+1)
        regret = best_arm_mean - hidden_dist[arm]
        total_regret[bandit] += regret
        cumulative_regret_history[bandit][t] = total_regret[bandit]
        reward = np.asscalar(bernoulli.rvs(hidden_dist[arm], size=1))
        bandit.update_arm(arm, reward)

#plot
for (key, value) in cumulative_regret_history.items():
    plt.plot(cumulative_regret_history[key], label=key_map[key])

plt.legend()
plt.show()







