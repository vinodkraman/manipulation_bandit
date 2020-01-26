from distributions.betadistribution import BetaDistribution
from bandits.bayesUCB import BayesUCB
from bandits.bayesUCBMean import BayesUCBMean
from scipy.stats import bernoulli
from bandits.random import Random
from natures.nature import Nature
import matplotlib.pyplot as plt
import numpy as np
from bandits.thompsonsampling import ThompsonSampling 
from noninfluencelimiters.noninfluencelimiter import NonInfluenceLimiter
from noninfluencelimiters.noninfluencelimiter2 import NonInfluenceLimiter2
from noninfluencelimiters.noninfluencelimiter3 import NonInfluenceLimiter3
from noninfluencelimiters.noninfluencelimiter4 import NonInfluenceLimiter4
from influencelimiters.influencelimiter import InfluenceLimiter
from influencelimiters.influencelimiter2 import InfluenceLimiter2
from influencelimiters.influencelimiter2test import InfluenceLimiter2test
from influencelimiters.influencelimiter2test3 import InfluenceLimiter2test3
from influencelimiters.influencelimiter2test4 import InfluenceLimiter2test4
from influencelimiters.influencelimiter2test5 import InfluenceLimiter2test5
from influencelimiters.influencelimiter2test7 import InfluenceLimiter2test7
from influencelimiters.influencelimiter2test6 import InfluenceLimiter2test6
from influencelimiters.influencelimiter2test8 import InfluenceLimiter2test8
from influencelimiters.influencelimiter2test9 import InfluenceLimiter2test9
from influencelimiters.influencelimiter2test10 import InfluenceLimiter2test10
from influencelimiters.influencelimiter2test11 import InfluenceLimiter2test11
from influencelimiters.influencelimiter2test12 import InfluenceLimiter2test12
from influencelimiters.influencelimiter2test13 import InfluenceLimiter2test13
from influencelimiters.influencelimiter2test14 import InfluenceLimiter2test14
from influencelimiters.influencelimiter2test16 import InfluenceLimiter2test16
from influencelimiters.influencelimiter2test17 import InfluenceLimiter2test17
from influencelimiters.influencelimiter2test18 import InfluenceLimiter2test18

from influencelimiters.influencelimiter2test15 import InfluenceLimiter2test15

from oracles.oracle import Oracle
from oracles.oracle2 import Oracle2
import scipy.stats
import copy
import progressbar

import sys
import yaml
import os
import time

YAML_FILE = "./config/params.yml"
with open(YAML_FILE, 'r') as file:
    y_d = yaml.load(file)

#####################HELPER FUNCTIONS#########################
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, 0), scipy.stats.sem(a, 0)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

pbar = progressbar.ProgressBar(redirect_stdout=True)
######################FIXED HYPERPARAMATERS###################
T = y_d["params"]["horizon"]                                               
K = y_d["params"]["no_arms"] 
num_exp = y_d["params"]["no_exp"] 
num_reports = y_d["params"]["no_reports"]  #control noise
trust_ratio = y_d["params"]["trust_ratio"] 
num_agents = y_d["params"]["no_agents"] 
attack = y_d["params"]["attack"] 
########################################################
trust = np.full(num_agents, False)
trust[:int(trust_ratio * num_agents)] = True
num_malicious  = num_agents - int(trust_ratio * num_agents)
c = 1
lam = np.log(num_malicious * c)
# trust = [True, False, False, True, True, False, False, False, False, True]
#####################END###############################
#scales with number of attackers, not fraction

world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
bayes_ucb_mean = BayesUCBMean(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)

il_ucb_21 = InfluenceLimiter2test4(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_213 = InfluenceLimiter2test3(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_215 = InfluenceLimiter2test5(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_214 = InfluenceLimiter2test4(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_217 = InfluenceLimiter2test7(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_216 = InfluenceLimiter2test6(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_218 = InfluenceLimiter2test8(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_219 = InfluenceLimiter2test9(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2110 = InfluenceLimiter2test10(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2111 = InfluenceLimiter2test11(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2112 = InfluenceLimiter2test12(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2113 = InfluenceLimiter2test13(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2114 = InfluenceLimiter2test14(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2115 = InfluenceLimiter2test15(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2116 = InfluenceLimiter2test16(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2117 = InfluenceLimiter2test17(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))
il_ucb_2118 = InfluenceLimiter2test18(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-1))

il_random = InfluenceLimiter2test14(copy.deepcopy(random), nature.agency, num_reports, np.exp(-1))

nil_b = NonInfluenceLimiter2(copy.deepcopy(bayes_ucb), nature.agency, num_reports)

oracle = Oracle2(copy.deepcopy(bayes_ucb), nature.agency)

# bandits = [il_ucb_12, bayes_ucb, nil_b, il_random]
# bandits = [il_ucb_10, il_ucb_11, il_ucb_13, il_ucb_15]
# bandits = [il_ucb_218, il_ucb_219, il_ucb_213, il_ucb_214, il_ucb_216, il_ucb_217, bayes_ucb, nil_b]
# bandits = [il_ucb_218, il_ucb_214, il_ucb_2110, bayes_ucb]
# bandits = [il_ucb_214, il_ucb_2111, il_ucb_2114, il_ucb_213, bayes_ucb]
bandits = [bayes_ucb, il_ucb_2115, il_ucb_2116, il_ucb_2117, il_ucb_2118]



key_map = {il_ucb_2115:"il_ucb_2115", il_ucb_2116:"il_ucb_2116", il_ucb_2117:"il_ucb_2117", il_ucb_2118:"il_ucb_2118", il_ucb_2110:"il_ucb_2110", il_ucb_2116:"il_ucb_2116", il_ucb_219:"il_ucb_219", il_ucb_217:"il_ucb_217", il_ucb_216: "il_ucb_216", il_ucb_215:"il_ucb_215", il_ucb_214: "il_ucb_214", il_ucb_213: "il_ucb_213", bayes_ucb: "bayes_ucb", il_random: "il_random", nil_b:"nil_b"}
key_color = {il_ucb_2115:"orange", il_ucb_2116:"black", il_ucb_2117:"red", il_ucb_2118:"blue", il_ucb_218:"green", il_ucb_2116:"brown", il_ucb_213:"green", il_ucb_214:"red", il_ucb_216:"brown", bayes_ucb: "purple", il_ucb_2110:"green" }

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

cumulative_trust_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_trust_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

for exp in pbar(range(num_exp)):
    #initialize arms
    nature.initialize_arms()

    #initialize trust order
    if attack == "copy":
        subset = trust[1:]
        np.random.shuffle(subset)
        trust[1:] = subset
    else:
        np.random.shuffle(trust)
    print(trust)

    #initialize agents
    nature.initialize_agents(trust, num_reports, y_d["params"]["no_targets"] )
    print("best arm: ", nature.best_arm)

    #reset bandits
    for bandit in bandits:
        bandit.reset()

    #reset oracle
    oracle.reset()

    print("hidden params:", nature.hidden_params)
    for t in range(T):
        # print("")
        # print("trust:", trust)
        # print("round:", t)
        rewards = nature.generate_rewards()
        # print("rewards", rewards)
        reports = nature.get_agent_reports(attack)
        # print(reports)
        oracle_arm = oracle.select_arm(t+1)
        
        oracle_reward = nature.generate_reward(oracle_arm)
        oracle.update(oracle_arm, oracle_reward)
        for bandit in bandits:
            arm = bandit.select_arm(t+1)
            # print("selected_arm:", arm)

            regret = nature.compute_per_round_regret(arm)
            # print("regret:", regret)
            oracle_regret = nature.compute_per_round_trust_regret(arm, oracle_arm)

            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]

            total_trust_regret[bandit][exp] += oracle_regret
            cumulative_trust_regret_history[bandit][exp][t] = total_trust_regret[bandit][exp]

            # reward = nature.generate_reward(arm)
            # print("reward:", reward)
            bandit.update(arm, rewards[arm])

            # if bandit == il_ucb_2111:
            #     # il_ucb_2111.plot_prediction_history(nature.best_arm)
            #     il_ucb_2111.plot_posterior_history(nature.best_arm)

    # il_ucb_2110.plot_reputations()
    # il_ucb_2111.plot_reputations()
    # il_ucb_214.plot_reputations()
    # il_ucb_2112.plot_reputations()
    # il_ucb_2115.plot_reputations()
    sys.stdout.flush()
    time.sleep(0.1)
    pbar.update(exp+1)

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
figure_title = ""
for (key, value) in average_cumulative_regret_history.items():
    plt.plot(average_cumulative_regret_history[key], label=key_map[key], color=key_color[key])
    figure_title += key_map[key] + "-"
    h = conf_cumulative_regret_history[key]
    plt.fill_between(range(T), average_cumulative_regret_history[key] - h, average_cumulative_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Mean Cumulative Regret")
plt.gca().set_ylim(bottom=0)
fig_path = "./figures/" + figure_title + "MCR.png"
plt.savefig(fig_path)
plt.clf()

figure_title = ""
for (key, value) in average_cumulative_trust_regret_history.items():
    plt.plot(average_cumulative_trust_regret_history[key], label=key_map[key], color=key_color[key])
    figure_title += key_map[key] + "-"
    h = conf_cumulative_trust_regret_history[key]
    plt.fill_between(range(T), average_cumulative_trust_regret_history[key] - h, average_cumulative_trust_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.xlabel("Round (t)")
plt.ylabel("Mean Cumulative Information Regret")
plt.gca().set_ylim(bottom=0)
fig_path = "./figures/" + figure_title + "MCIR.png"
plt.savefig(fig_path)
plt.clf()