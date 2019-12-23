from BetaDistribution import BetaDistribution
from BayesUCB import BayesUCB
from scipy.stats import bernoulli
from Random import Random
from Nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from ThompsonSampling import ThompsonSampling 
from NonInfluenceLimiter import NonInfluenceLimiter
from InfluenceLimiter import InfluenceLimiter
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
trust = [True, False]
initial_reputations = 1

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)
nil = NonInfluenceLimiter(copy.copy(bayes_ucb), nature.agency, num_reports)
il = InfluenceLimiter(copy.copy(bayes_ucb), nature.agency, num_reports)
# bandits = [thompson, bayes_ucb, random, nil]
bandits = [il, nil]
# bandits = [nil]

key_map = {thompson: "thompson", bayes_ucb: "bayes_ucb", random: "random", nil: "nil", il: "il"}
key_color = {thompson: "red", bayes_ucb: "blue", random: "gray", nil: "green", il: "orange"}

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

#run bandit
for bandit in bandits:
    for exp in range(num_exp):
        #reset
        nature.initialize_arms()
        if bandit == nil or bandit == il:
            # print(bandit)
            nature.initialize_agents(trust, num_reports, initial_reputations)
            np.random.shuffle(trust)
        
        # [[print(dist.get_params()) for dist in agent.arm_dists] for agent in nature.agency.agents]
        bandit.reset()

        for t in range(T):
            # print("round")
            reports = nature.get_agent_reports()
            # print(reports)
            arm = bandit.select_arm(t+1)
            # print("selected_arm", arm)
            regret = nature.compute_per_round_regret(arm)
            # print("regret", regret)
            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]/(t+1)
            reward = nature.generate_reward(arm)
            # print("reward", reward)
            bandit.update(arm, reward)

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
plt.ylabel("Mean Cumulative Regret/t")
plt.ylim(0, 1)
plt.show()