from BetaDistribution import BetaDistribution
from BayesUCB import BayesUCB
from scipy.stats import bernoulli
from Random import Random
from Nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from ThompsonSampling import ThompsonSampling 

T = 2000
K = 10
num_exp = 10

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors)

bayes_ucb = BayesUCB(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)
bandits = [thompson, bayes_ucb, random]

key_map = {thompson: "thompson", bayes_ucb: "bayes_ucb", random: "random"}

cumulative_regret_history = {bandit: {exp:np.zeros(T) for exp in range(num_exp)} for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

#run bandit
for bandit in bandits:
    for exp in range(num_exp):
        #reset
        nature.initialize_arms()
        bandit.reset()
        for t in range(T):
                arm = bandit.select_arm(t+1)
                regret = nature.compute_per_round_regret(arm)
                total_regret[bandit][exp] += regret
                cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]/(t+1)
                reward = nature.generate_reward(arm)
                bandit.update_arm(arm, reward)

#average over experiments
average_cumulative_regret_history = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_regret_history.items():
    sum_regret = np.zeros(T)
    for (experiment_num, regret) in experiments.items():
        sum_regret += regret

    sum_regret /= num_exp
    average_cumulative_regret_history[bandit] = sum_regret

#plot
for (key, value) in average_cumulative_regret_history.items():
    plt.plot(average_cumulative_regret_history[key], label=key_map[key])

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Cumulative Regret/t")
plt.show()