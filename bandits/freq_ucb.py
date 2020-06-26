from bandits.bandit import Bandit
import numpy as np
from random import *
import operator

#the idea is to use expert advice to eliminate arms that we know are bad. Problem is, in the begininning, we cant immediatley follow expert advice, so we instead randomly selecet

class Freq_UCB(Bandit):

    def compute_KLD(self, p, q):
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        else: 
            # print(q)
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def select_arm(self, t, q_tilde, reports, W):
        if t <= 1*len(self.arms):
            return (t-1)%self.K

        ucb = [arm.mean_reward() + np.sqrt(2*np.log(t)/(arm.pulls)) for (index, arm) in enumerate(self.arms)]
        lcb = [arm.mean_reward() - np.sqrt(2*np.log(t)/(arm.pulls)) for (index, arm) in enumerate(self.arms)]
        mean_estimate = np.array([(arm.rewards + q_tilde[index]*reports[index])/(arm.pulls + reports[index]) for (index, arm) in enumerate(self.arms)])
        exploration_bonus = np.array([np.sqrt(2*np.log(t)/(arm.pulls + reports[index])) for (index, arm) in enumerate(self.arms)])

        clipped = np.clip(mean_estimate, lcb, ucb)
        return np.argmax(clipped)



        # exploration_bonus = [np.sqrt(2*np.log(t)/(arm.pulls + reports[index])) for (index, arm) in enumerate(self.arms)]

        # print(exploration_bonus)
        # means = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        # mean_estimate = [(arm.rewards + q_tilde[index]*reports[index])/(arm.pulls + reports[index]) for (index, arm) in enumerate(self.arms)]

        # final = [min(means[index] + exploration_bonus[index], mean_estimate) for (index, arm) in enumerate(self.arms)]

        # print(mean_estimate)

        # if W <= 1:
        #     ucb = np.array(mean_estimate) + np.array(exploration_bonus)
        # else:
        #     ucb = np.array(mean_estimate) - np.array(exploration_bonus)


        # print(ucb)
        # return np.argmax(ucb)
        #compute new average: sum the rewards p
        # print("q_tilde:", q_tilde)
        # print("reports:", reports)
        # exploration_bonus = [np.sqrt(2*np.log(t)/(arm.pulls + reports[index])) for (index, arm) in enumerate(self.arms)]
        # mean_estimate = [(arm.rewards + q_tilde[index]*reports[index])/(arm.pulls + reports[index]) for (index, arm) in enumerate(self.arms)]
        # pure_mean_estimate = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        # # print("exploration_bonus:", exploration_bonus)
        # # print("mean_estimate:", mean_estimate)
        # pulls = [arm.pulls for arm in self.arms]
        # arg_sorted = np.argsort(pulls)
        # print("pulls",pulls)
        # print(max(pulls)/np.)
        # test = [W[self.arms[index]] for index in range(self.K)]
        # print(test)

        # if t < 500:
        #     ucb = np.array(pure_mean_estimate)
        # else:
        # if W <= 1:
        #     exploration_bonus = [np.sqrt(2*np.log(t)/(arm.pulls)) for (index, arm) in enumerate(self.arms)]
        #     mean_estimate = [(arm.rewards)/(arm.pulls) for (index, arm) in enumerate(self.arms)]
        #     ucb = np.array(mean_estimate) + np.array(exploration_bonus)
        # else:
        #     # mean_estimate = [(arm.rewards + q_tilde[index]*reports[index])/(arm.pulls + reports[index]) for (index, arm) in enumerate(self.arms)]
        #     # ucb = mean_estimate
        #     exploration_bonus = [np.sqrt(2*np.log(t)/(arm.pulls + min(reports[index], 30))) for (index, arm) in enumerate(self.arms)]
        #     mean_estimate = [(arm.rewards + q_tilde[index]*min(reports[index], 30))/(arm.pulls + min(reports[index], 30)) for (index, arm) in enumerate(self.arms)]
        #     ucb = np.array(mean_estimate) + np.array(exploration_bonus)

        # return np.argmax(ucb)
            

    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
