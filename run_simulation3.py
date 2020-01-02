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
from NonInfluenceLimiter4 import NonInfluenceLimiter4
from InfluenceLimiter import InfluenceLimiter
from InfluenceLimiter2 import InfluenceLimiter2
from InfluenceLimiter3 import InfluenceLimiter3
from Oracle import Oracle
from Oracle3 import Oracle3
import scipy.stats
import copy

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

######################HYPERPARAMATERS####################
T = 500                                                 
K = 10            
num_exp = 25
num_reports = 10 #control noise
trust_ratio = 0.30
num_agents = 10
trust = np.full(num_agents, False)
trust[:int(trust_ratio * num_agents)] = True
# trust = [False, False, False, False, True]
#####################END###############################


world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)

il_rep_0 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(0))
proposed_il_ucb = InfluenceLimiter(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_rep_2 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-2))
il_rep_5 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-5))

old_il_ucb = InfluenceLimiter3(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_rep_00 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, 0)

il_ucb_1 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_11 = InfluenceLimiter(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))

il_ucb_3 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-3))
il_ucb_5 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-5))
il_ucb_01 = InfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(0))
il_thom_1 = InfluenceLimiter2(copy.deepcopy(thompson), nature.agency, num_reports, np.exp(-1))

# proposed_il_ucb = il_rep_1

il_random = InfluenceLimiter(copy.deepcopy(random), nature.agency, num_reports, np.exp(-1))

nil_c = NonInfluenceLimiter4(copy.deepcopy(bayes_ucb), nature.agency, 0.50, num_reports)
nil_b = NonInfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports)

oracle = Oracle(copy.deepcopy(bayes_ucb), nature.agency)
oracle_test = Oracle(copy.deepcopy(bayes_ucb), nature.agency)

# bandits = [bayes_ucb, il, nil2]
# bandits = [bayes_ucb, il_rep_1, il_random, nil]
# bandits = [proposed_il_ucb, old_il_ucb, bayes_ucb, il_random]
# bandits = [il_ucb_1, il_ucb_3, il_ucb_5, il_ucb_01, bayes_ucb, il_rep_00]
bandits = [il_ucb_1, bayes_ucb]
# bandits = [il_ucb]

# key_map = {bayes_ucb: "bayes_ucb", il_rep_0: "il_rep_0", proposed_il_ucb: "proposed_il_ucb", old_il_ucb: "old_il_ucb", il_rep_5: "il_rep_5", oracle_test:"oracle_test", nil_b:"nil_b", nil_c:"nil_c", il_random:"il_random", updated_proposed_il_ucb:"updated_proposed_il_ucb"}
# key_color = {il_rep_0: "red", proposed_il_ucb: "green", old_il_ucb: "blue", il_rep_5: "yellow", bayes_ucb:"purple", oracle_test:"purple", nil_b:"orange", nil_c:"black", il_random:"orange", updated_proposed_il_ucb:"orange"}

# key_map = {il_ucb_1: "initial rep = e^-1", il_ucb_3: "initial rep = e^-3", il_ucb_5: "initial rep = e^-5", il_ucb_01: "initial rep = 1", bayes_ucb: "bayes_ucb", il_rep_00:"initial rep = 0"}
# key_color = {il_ucb_1: "red", il_ucb_3: "blue", il_ucb_5: "green", il_ucb_01: "orange", bayes_ucb: "purple", il_rep_00: "black"}
key_map = {il_ucb_11: "il_thil_ucb_11om_1", il_ucb_1:"il_ucb_1", bayes_ucb:"bayes_ucb", thompson: "thompson"}
key_color = {il_ucb_11: "red", il_ucb_1: "blue", bayes_ucb: "green", thompson: "purple"}

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

cumulative_trust_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_trust_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

for exp in range(num_exp):
    #initialize arms
    nature.initialize_arms()

    #initialize trust order
    np.random.shuffle(trust)

    #initialize agents
    nature.initialize_agents(trust, num_reports)

    #reset bandits
    for bandit in bandits:
        bandit.reset()

    #reset oracle
    oracle.reset()
    for t in range(T):
        # nature.agency.track_reputations()
        reports = nature.get_agent_reports()
        oracle_arm = oracle.select_arm(t+1)
        
        oracle_reward = nature.generate_reward(oracle_arm)
        oracle.update(oracle_arm, oracle_reward)
        for bandit in bandits:
            
            # print("bandit", bandit)
            # if t == 0:
            #     print("pre")
            #     for test in bandit.bandit.arms:
            #         print("alpha reward:", test.reward_dist.alpha)
            #         print("beta reward:", test.reward_dist.beta)
            #         print("alpha influence:", test.influence_reward_dist.alpha)
            #         print("beta influence:", test.influence_reward_dist.beta)
                
            arm = bandit.select_arm(t+1)
            # if t == 0:
            #     print("post")
            #     for test in bandit.bandit.arms:
            #         print("alpha reward:", test.reward_dist.alpha)
            #         print("beta reward:", test.reward_dist.beta)
            #         print("alpha influence:", test.influence_reward_dist.alpha)
            #         print("beta influence:", test.influence_reward_dist.beta)

            regret = nature.compute_per_round_regret(arm)
            oracle_regret = nature.compute_per_round_trust_regret(arm, oracle_arm)

            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]/(t+1)

            total_trust_regret[bandit][exp] += oracle_regret
            cumulative_trust_regret_history[bandit][exp][t] = total_trust_regret[bandit][exp]/(t+1)

            reward = nature.generate_reward(arm)
            bandit.update(arm, reward)

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