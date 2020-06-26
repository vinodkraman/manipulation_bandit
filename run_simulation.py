from distributions.betadistribution import BetaDistribution
from bandits.bayesUCB import BayesUCB
from bandits.UCB_influence import UCB_influence
from bandits.UCB_influence_2 import UCB_influence_2
from bandits.freq_ucb import Freq_UCB
from bandits.thompson_sampling import Thompson_Sampling
from bandits.thompson_sampling_inf import Thompson_Sampling_Inf


from bandits.UCB1 import UCB1
from bandits.bayesUCBMean import BayesUCBMean
from scipy.stats import bernoulli
from bandits.random import Random
from bandits.expert_greedy import Expert_greedy
from bandits.ucb_expert_greedy import UCB_Expert_Greedy
from bandits.KL_expert_greedy import KL_Expert_Greedy
from bandits.epsilongreedy import EpsilonGreedy
from natures.nature import Nature
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from bandits.thompsonsampling import ThompsonSampling 
import seaborn as sns


sns.set()
sns.set_style("white")
sns.set_style("ticks")
matplotlib.rcParams.update({'font.size': 30})


from noninfluencelimiters.noninfluencelimiter import NonInfluenceLimiter
from noninfluencelimiters.noninfluencelimiter_ucb import NonInfluenceLimiter_UCB
from noninfluencelimiters.non_exp5 import Non_exp5

from influencelimiters.influencelimiter_study_2 import InfluenceLimiter_study_2
from influencelimiters.influencelimiter_study_3 import InfluenceLimiter_study_3
from influencelimiters.influencelimiter_study_5 import InfluenceLimiter_study_5
from influencelimiters.influencelimiter_study_6 import InfluenceLimiter_study_6
from influencelimiters.influencelimiter_study_7 import InfluenceLimiter_study_7
from influencelimiters.influencelimiter_study_8 import InfluenceLimiter_study_8
from influencelimiters.influencelimiter_study_9 import InfluenceLimiter_study_9
from influencelimiters.influencelimiter_study_10 import InfluenceLimiter_study_10
# from influencelimiters.influencelimiter_study_11 import InfluenceLimiter_study_11
from influencelimiters.influencelimiter_study_12 import InfluenceLimiter_study_12
from influencelimiters.influencelimiter_study_13 import InfluenceLimiter_study_13
from influencelimiters.influencelimiter_study_14 import InfluenceLimiter_study_14
from influencelimiters.influencelimiter_freq import Influencelimiter_freq
from influencelimiters.influencelimiter_freq_test import Influencelimiter_freq_test
from influencelimiters.influencelimiter_bayes import InfluenceLimiter_bayes
from influencelimiters.influencelimiter_frequent import Influencelimiter_frequent


from influencelimiters.exp5 import Exp5
from influencelimiters.exp5_test import Exp5_test


from oracles.oracle import Oracle
from oracles.oracle2 import Oracle2
from oracles.oracle_ucb import Oracle_ucb

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
attack_freq = y_d["params"]["attack_freq"]
########################################################
trust = np.full(num_agents, False)
trust[:int(trust_ratio * num_agents)] = True
num_malicious  = num_agents - int(trust_ratio * num_agents)
c = 1
lam = np.log(num_malicious * c)
#####################END###############################
world_priors = [BetaDistribution(1, 1) for k in range(K)]
nature = Nature(K, world_priors, len(trust))

bayes_ucb = BayesUCB(T, K, world_priors)
bayes_ucb_mean = BayesUCBMean(T, K, world_priors)
random = Random(T, K, world_priors)
thompson = ThompsonSampling(T, K, world_priors)
ucb_influence = UCB_influence(T, K, world_priors)
ucb = UCB1(T, K, world_priors)
ucb_influence_2 = UCB_influence_2(T, K, world_priors)
expert_greedy = Expert_greedy(T, K, world_priors)
ucb_expert_greedy = UCB_Expert_Greedy(T, K, world_priors)
kl_expert_greedy = KL_Expert_Greedy(T, K, world_priors)
epsilon_greedy = EpsilonGreedy(T, K, world_priors)
freq_ucb = Freq_UCB(T, K, world_priors)
thompson_sampling = Thompson_Sampling(T, K, world_priors)
thompson_sampling_inf = Thompson_Sampling_Inf(T, K, world_priors)


influencelimiter_ucb = InfluenceLimiter_study_10(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-6))
influencelimiter_thompson = InfluenceLimiter_study_10(copy.deepcopy(thompson), nature.agency, num_reports, np.exp(-6))
ucb_influencelimitier_1 = Influencelimiter_freq(copy.deepcopy(ucb_influence), nature.agency, num_reports, np.exp(-10), 100)
ucb_influencelimitier_10 = Influencelimiter_freq(copy.deepcopy(kl_expert_greedy), nature.agency, num_reports, np.exp(-20), 50)
ucb_influencelimitier_20 = Influencelimiter_freq(copy.deepcopy(expert_greedy), nature.agency, num_reports, np.exp(-10), 50)
ucb_influencelimitier_50 = Influencelimiter_freq(copy.deepcopy(expert_greedy), nature.agency, num_reports, np.exp(-10), 50)
greedy_influencelimitier = Influencelimiter_freq_test(copy.deepcopy(kl_expert_greedy), nature.agency, num_reports, np.exp(-5), 50)
test_influencelimitier = Influencelimiter_frequent(copy.deepcopy(kl_expert_greedy), nature.agency, num_reports, np.exp(-10), 50)
bayes_influence_limiter = InfluenceLimiter_bayes(copy.deepcopy(bayes_ucb), nature.agency, num_reports, np.exp(-15))
thom_influence_limiter = InfluenceLimiter_bayes(copy.deepcopy(thompson_sampling_inf), nature.agency, num_reports, np.exp(-15))

exp5 = Exp5(copy.deepcopy(ucb_influence), nature.agency, num_reports, np.exp(-5))
freq_influencelimiter = Influencelimiter_frequent(copy.deepcopy(freq_ucb), nature.agency, num_reports, np.exp(-15), 50)

non_influencelimiter = NonInfluenceLimiter(copy.deepcopy(bayes_ucb), nature.agency, num_reports)
non_exp5 = Non_exp5(copy.deepcopy(bayes_ucb), nature.agency, num_reports)
non_influencelimiter_ucb = NonInfluenceLimiter_UCB(copy.deepcopy(freq_ucb), nature.agency, num_reports, 50)

oracle = Oracle_ucb(copy.deepcopy(expert_greedy), nature.agency, 50)

bandits = [thompson_sampling, thom_influence_limiter]

key_map = {
                ucb_influencelimitier_1:"ucb_influencelimitier_1",
                ucb_influencelimitier_10:"ucb_influencelimitier_10",
                ucb_influencelimitier_20:"ucb_influencelimitier_20",
                ucb_influencelimitier_50:"ucb_influencelimitier_50", 
                ucb:"ucb",
                greedy_influencelimitier:"greedy_influencelimitier",
                non_influencelimiter_ucb:"non_influencelimiter_ucb",
                influencelimiter_ucb:"bayes_ucb",
                test_influencelimitier:"influencelimitier",
                bayes_influence_limiter:"bayes_influence_limiter",
                bayes_ucb:"bayes_ucb",
                freq_influencelimiter:"freq_influencelimiter",
                thom_influence_limiter:"thom_influence_limiter",
                thompson_sampling:"thompson_sampling"
}
# key_map = {non_influencelimiter_ucb:"non_influencelimiter_ucb", ucb_influencelimitier_2:"ucb_influencelimitier_2", non_exp5:"non_exp5", ucb:"ucb", exp5:"SEQ-EXP", ucb_influencelimitier:"ucb_influencelimitier", influencelimiter_ucb:"influencelimiter_ucb", non_influencelimiter:"non_influencelimiter", bayes_ucb:"bayes_ucb", thompson:"thompson"}
key_color = {
                ucb_influencelimitier_1:"red", 
                ucb_influencelimitier_10:"purple", 
                ucb_influencelimitier_20:"black", 
                ucb_influencelimitier_50:"green", 
                exp5:"red", 
                ucb: "orange",
                non_influencelimiter_ucb:"pink",
                greedy_influencelimitier:"brown",
                influencelimiter_ucb:"green",
                test_influencelimitier:"blue",
                bayes_influence_limiter:"blue",
                bayes_ucb:"green",
                freq_influencelimiter:"red",
                thom_influence_limiter:"purple",
                thompson_sampling:"red"
}

cumulative_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

cumulative_trust_regret_history = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_trust_regret = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

cumulative_reward = {bandit: np.zeros((num_exp, T)) for bandit in bandits}
total_reward = {bandit: {exp:0 for exp in range(num_exp)} for bandit in bandits}

for exp in pbar(range(num_exp)):
    #initialize arms
    nature.initialize_arms()
    nature.initialize_agents(trust, num_reports, y_d["params"]["no_targets"], attack_freq)

    #reset bandits
    for bandit in bandits:
        bandit.reset()

    # reset oracle
    oracle.reset()
    # print("hidden params:", nature.hidden_params)
    # print("best arm:", nature.best_arm)
    for t in range(T):
        # print("")
        # print("round:", t)
        # print("hidden params:", nature.hidden_params)
        # print("best arm:", nature.best_arm)
        # print("worst arm:", nature.worst_arm)
        nature.shuffle_agents()

        # print("agents:", nature.agency.agents)

        # order = []
        # [order.append(agent.id) for agent in nature.agency.agents]
        # print("agent order:", order)

        # trust_vec = []
        # [trust_vec.append(agent.trustworthy) for agent in nature.agency.agents]
        # print("agent trust:", trust_vec)

        rewards = nature.generate_rewards(2)
        # print("rewards:", rewards)

        reports = nature.get_agent_reports(t, attack)
        # [print(reports[agent]) for agent in nature.agency.agents]

        # oracle_arm, pred = oracle.select_arm(t+1)
        # oracle_reward = rewards[oracle_arm]
        # oracle.update(oracle_arm, oracle_reward)
        
        for bandit in bandits:
            arm, prediction = bandit.select_arm(t+1)
            # print("selected_arm:", arm)

            regret = nature.compute_per_round_regret(arm)
            # oracle_regret = nature.compute_per_round_trust_regret(arm, oracle_arm)

            total_regret[bandit][exp] += regret
            cumulative_regret_history[bandit][exp][t] = total_regret[bandit][exp]

            # total_trust_regret[bandit][exp] += oracle_regret
            # cumulative_trust_regret_history[bandit][exp][t] = total_trust_regret[bandit][exp]

            total_reward[bandit][exp] += np.max(rewards) - rewards[arm]
            cumulative_reward[bandit][exp][t] = total_reward[bandit][exp]

            bandit.update(arm, rewards[arm])


    # exp5.plot_reputations()
    # plt.plot(cumulative_regret_history[ucb_influencelimitier][exp])
    # thom_influence_limiter.plot_reputations()
    # plt.plot(cumulative_regret_history[thom_influence_limiter][exp], color="red")
    # plt.plot(cumulative_regret_history[thompson_sampling][exp], color="blue")
    # # # # # plt.plot(cumulative_reward[test_influencelimitier][exp])
    # plt.show()

    # ucb_influencelimitier_1.plot_reputations()
    # plt.plot(cumulative_regret_history[ucb_influencelimitier_1][exp])
    # plt.show()

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


average_cumulative_reward = {i:np.zeros(T) for i in bandits}
conf_cumulative_reward = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_reward.items():
    mean, conf = mean_confidence_interval(experiments)
    average_cumulative_reward[bandit] = mean
    conf_cumulative_reward[bandit] = conf

average_cumulative_trust_regret_history = {i:np.zeros(T) for i in bandits}
conf_cumulative_trust_regret_history = {i:np.zeros(T) for i in bandits}
for (bandit, experiments) in cumulative_trust_regret_history.items():
    mean, conf = mean_confidence_interval(experiments)
    average_cumulative_trust_regret_history[bandit] = mean
    conf_cumulative_trust_regret_history[bandit] = conf

#plot
figure_title = ""
# plt.figure(figsize=(9,5)) 
plt.figure(figsize=(7,6)) 
for (key, value) in average_cumulative_regret_history.items():
    plt.plot(average_cumulative_regret_history[key], label=key_map[key], color=key_color[key])
    figure_title += key_map[key] + "-"
    h = conf_cumulative_regret_history[key]
    plt.fill_between(range(T), average_cumulative_regret_history[key] - h, average_cumulative_regret_history[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.tick_params(labelsize=15)
plt.xlabel("Round", fontsize=15)
plt.ylabel("Mean Cumulative Regret", fontsize=15)
plt.gca().set_ylim(bottom=0)
fig_path = "./figures/" + figure_title + "MCR.png"
plt.savefig(fig_path)
plt.clf()

figure_title = ""
# plt.figure(figsize=(9,5)) 
plt.figure(figsize=(7,6)) 
for (key, value) in average_cumulative_reward.items():
    plt.plot(average_cumulative_reward[key], label=key_map[key], color=key_color[key])
    figure_title += key_map[key] + "-"
    h = conf_cumulative_reward[key]
    plt.fill_between(range(T), average_cumulative_reward[key] - h, average_cumulative_reward[key] + h,
                 color=key_color[key], alpha=0.2)

plt.legend()
plt.tick_params(labelsize=15)
plt.xlabel("Round", fontsize=15)
plt.ylabel("Mean Cumulative Reward", fontsize=15)
plt.gca().set_ylim(bottom=0)
fig_path = "./figures/" + figure_title + "MCReward.png"
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