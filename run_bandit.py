from BetaDistribution import BetaDistribution
from BayesUCB import BayesUCB
from scipy.stats import bernoulli
from Random import Random
from Nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from ThompsonSampling import ThompsonSampling 
from Oracle import Oracle
from Oracle4 import Oracle4
import scipy.stats
import copy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

T = 500
K = 5
num_exp = 10
num_reports = 10
trust = [False, False, True, False]
initial_reputations = 1

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)
oracle = Oracle4(copy.deepcopy(bayes_ucb), nature.agency)
bandits = [thompson, bayes_ucb, random]

key_map = {thompson: "Thompson", bayes_ucb: "Bayes UCB", random: "Random"}
key_color = {thompson: "red", bayes_ucb: "blue", random: "green"}

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

#run bandit
for bandit in bandits:
    for exp in range(num_exp):
        #reset
        nature.initialize_arms()
        nature.initialize_agents(trust, num_reports)        
        bandit.reset()
        oracle.reset()
        for t in range(T):
            report = nature.get_agent_reports()
            arm = bandit.select_arm(t+1)
            oracle_arm = oracle.select_arm(t+1)

            regret = nature.compute_per_round_regret(arm)
            regret = nature.compute_per_round_trust_regret(arm, oracle_arm)

            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]/(t+1)

            reward = nature.generate_reward(arm)
            oracle_reward = nature.generate_reward(oracle_arm)

            bandit.update(arm, reward)
            oracle.update(oracle_arm, oracle_reward)

#average over experiments
average_cumulative_regret_history = {i:np.zeros(T) for i in bandits}
conf_cumulative_regret_history = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_regret_history.items():
    mean, conf = mean_confidence_interval(experiments)
    average_cumulative_regret_history[bandit] = mean
    conf_cumulative_regret_history[bandit] = conf

#plot
for (key, value) in average_cumulative_regret_history.items():
    plt.plot(average_cumulative_regret_history[key], label=key_map[key], color=key_color[key])
    h = conf_cumulative_regret_history[key]
    plt.fill_between(range(T), average_cumulative_regret_history[key] - h, average_cumulative_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Mean Cumulative Information Regret")
plt.ylim(0, 1)
plt.show()