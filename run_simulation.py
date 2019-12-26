from BetaDistribution import BetaDistribution
from BayesUCB import BayesUCB
from scipy.stats import bernoulli
from Random import Random
from Nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from ThompsonSampling import ThompsonSampling 
from NonInfluenceLimiter import NonInfluenceLimiter
from NonInfluenceLimiter2 import NonInfluenceLimiter2
from NonInfluenceLimiter3 import NonInfluenceLimiter3
from InfluenceLimiter import InfluenceLimiter
from InfluenceLimiter2 import InfluenceLimiter2
from Oracle import Oracle
from Oracle2 import Oracle2
from Oracle3 import Oracle3
from Oracle4 import Oracle4
import scipy.stats
import copy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

T = 100
K = 5
num_exp = 25
num_reports = 10
# trust = [False, True, False, False, True]
trust = [False, True, False, True, False]
initial_reputations = np.exp(-1)

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)
nil = NonInfluenceLimiter(copy.copy(bayes_ucb), nature.agency, num_reports)
nil2 = NonInfluenceLimiter2(copy.copy(bayes_ucb), nature.agency, num_reports)
nil3 = NonInfluenceLimiter3(copy.copy(bayes_ucb), nature.agency, num_reports)
il = InfluenceLimiter(copy.copy(bayes_ucb), nature.agency, num_reports)
il2 = InfluenceLimiter2(copy.copy(bayes_ucb), nature.agency, num_reports)
oracle = Oracle3(copy.copy(bayes_ucb), nature.agency)

bandits = [random, bayes_ucb, il, nil2]

key_map = {thompson: "thompson", bayes_ucb: "bayes_ucb", random: "random", nil: "nil", il: "il", nil2: "nil2", il2:"il2", nil3:"nil3"}
key_color = {thompson: "red", bayes_ucb: "blue", random: "gray", nil: "green", il: "orange", nil2: "green", il2:"red", nil3:"green"}

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

cumulative_trust_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_trust_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

#run bandit
for bandit in bandits:
    for exp in range(num_exp):
        #reset
        nature.initialize_arms()
        nature.initialize_agents(trust, num_reports, initial_reputations)

        if bandit == il or bandit == il2 or bandit == nil3:
            np.random.shuffle(trust)
        
        bandit.reset()
        # print("best arm:", nature.best_arm)
        oracle.reset()
        for t in range(T):
            # print(len(nature.agency.agents))
            # print("round", t)
            # nature.agency.track_reputations()
            reports = nature.get_agent_reports()
            arm = bandit.select_arm(t+1)
            # print("selected arm", arm)
            oracle_arm = oracle.select_arm(t+1)

            regret = nature.compute_per_round_regret(arm)
            # print("regret:", regret)
            oracle_regret = nature.compute_per_round_trust_regret(arm, oracle_arm)

            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]/(t+1)

            total_trust_regret[bandit][exp] += oracle_regret
            cumulative_trust_regret_history[bandit][exp][t] = total_trust_regret[bandit][exp]/(t+1)

            reward = nature.generate_reward(arm)
            # print("reward:", reward)
            oracle_reward = nature.generate_reward(oracle_arm)

            bandit.update(arm, reward)
            oracle.update(arm, reward)

        # nature.agency.plot_reputations()
# average over experiments
average_cumulative_regret_history = {i:np.zeros(T) for i in bandits}
conf_cumulative_regret_history = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_regret_history.items():
    mean, conf = mean_confidence_interval(experiments)
    average_cumulative_regret_history[bandit] = mean
    conf_cumulative_regret_history[bandit] = conf

average_cumulative_trust_regret_history = {i:np.zeros(T) for i in bandits}
conf_cumulative_trust_regret_history = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_trust_regret_history.items():
    mean, conf = mean_confidence_interval(experiments)
    average_cumulative_trust_regret_history[bandit] = mean
    conf_cumulative_trust_regret_history[bandit] = conf

#plot
for (key, value) in average_cumulative_regret_history.items():
    plt.plot(average_cumulative_regret_history[key], label=key_map[key], color=key_color[key])
    h = conf_cumulative_regret_history[key]
    plt.fill_between(range(T), average_cumulative_regret_history[key] - h, average_cumulative_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Mean Cumulative Regret")
plt.ylim(0, 1)
plt.show()

for (key, value) in average_cumulative_trust_regret_history.items():
    plt.plot(average_cumulative_trust_regret_history[key], label=key_map[key], color=key_color[key])
    h = conf_cumulative_trust_regret_history[key]
    plt.fill_between(range(T), average_cumulative_trust_regret_history[key] - h, average_cumulative_trust_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Mean Cumulative Information Regret")
plt.ylim(0, 1)
plt.show()