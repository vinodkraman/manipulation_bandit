from bandits.bandit import Bandit
import numpy as np
from random import *
import operator

#the idea is to use expert advice to eliminate arms that we know are bad. Problem is, in the begininning, we cant immediatley follow expert advice, so we instead randomly selecet

class Expert_greedy(Bandit):

    def compute_KLD(self, p, q):
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        else: 
            # print(q)
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    #problem we face here, is that we pull the good arm fast, and then we don't pull the other arm fast enough 
    def select_arm(self, t, q_tilde, W, C, order= None):        
        if t <= 1*len(self.arms):
            return (t-1)%self.K, (t-1)%self.K

        val_mean = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        val_ucb = [arm.mean_reward() + np.sqrt(2*np.log(t)/arm.pulls) for (index, arm) in enumerate(self.arms)]
        val_arm_pulls = np.array([arm.real_pulls for (index, arm) in enumerate(self.arms)]) + 1

        # print(val_arm_pulls)
        bonus = (val_arm_pulls/np.sum(val_arm_pulls))
        ucb_plus_bonus = bonus + np.array(val_ucb)

        # print(val_arm_pulls)
        return np.argmax(ucb_plus_bonus), np.argmax(q_tilde)
        # filtered_vals = self.filter_arms(val_arm_pulls, t)
        # filtered_vals = np.argsort(val_arm_pulls)
        # filtered_vals = filtered_vals[-3:]
        # print(val_arm_pulls)
        # return np.argmax(val_ucb), np.argmax(q_tilde)
        # print(val_mean)
        # print(filtered_vals)
        # expert_arm = np.argmax(q_tilde)
        # edge = val_arm_pulls[expert_arm]/t
        # print(edge)
        # probs = [(1-edge)/self.K for arm in self.arms]
        # probs[expert_arm] += edge
        # norm = val_arm_pulls/np.sum(val_arm_pulls)
        # # print(norm)
        # should_explore = np.random.rand()
        # if should_explore < 1:
        #     return np.random.choice(self.K, 1, p=norm)[0], np.argmax(q_tilde)
        # else:
        #     return np.argmax(val_mean), np.argmax(q_tilde)
        # weighted = norm + np.array(val_ucb)
        # # print(weighted)
        # return np.argmax(weighted), np.argmax(q_tilde)

        expert_arm = np.argmax(q_tilde)
        edge = 0.75
        probs = [(1-edge)/self.K for arm in self.arms]
        probs[expert_arm] += edge

        if should_explore < 0.10:
            return np.random.choice(self.K, 1, p=probs)[0], np.argmax(q_tilde)
        else:
            return np.argmax(val_mean), np.argmax(q_tilde)

        rand_vec = np.random.rand(self.K)
        arm_set = [index for (index, value) in enumerate(probs) if value < rand_vec[index]]

        means = [val_ucb[index] for index in arm_set]
        index_index = np.argmax(means)

        return arm_set[index_index], np.argmax(q_tilde)


        # return np.argmax(val_ucb), np.argmax(q_tilde)

        means = [val_ucb[index] for index in filtered_vals]
        index_index = np.argmax(means)

        means_q = [q_tilde[index] for index in filtered_vals]
        index_q = np.argmax(means_q)

        return filtered_vals[index_index], np.argmax(q_tilde)

        # # randnums = set(order[-1000:])
        # # print(randnums)
        # val_mean = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        # # print("Means", val_mean)
        # val = [q_tilde[index] for (index, arm) in enumerate(self.arms)]

        # val_ucb = [arm.mean_reward() + np.sqrt(2*np.log(t)/arm.pulls) for (index, arm) in enumerate(self.arms)]

        # val_test = [q_tilde[index] + np.sqrt(2*np.log(t)/arm.pulls) for (index, arm) in enumerate(self.arms)]

        # # return np.argmax(val_test), 0.50
        # # # print("Predictions", q_tilde)
        # val = [arm.real_pulls for (index, arm) in enumerate(self.arms)]

        # # pick the arm 
        # #remove arms so that 95 percent of energy is left
        # print(val)
        # # # print(val_mean)
        # print(self.filter_arms(val, t))
        # kuku = np.argsort(val)
        # # print(kuku[-3:])
        # kuku_3 = kuku[-3:]
        # selected_arm = np.argmax(q_tilde)
        # should_explore = np.random.rand()

        # filtered_vals = self.filter_arms(val, t)
        # # if len(filtered_vals) == 1:
        # #     return filtered_vals[0], 1
        # #1  - 1/np.log(np.exp(1) + t)
        # kii = t/(t+50)
        # # print(kii)
        # if should_explore < 1:
        #     means = [val_ucb[index] for index in filtered_vals]
        #     index_index = np.argmax(means)
        #     return filtered_vals[index_index], selected_arm
        #     # return np.argmax(val_ucb), 1
        # else:
        #     return selected_arm, 2
        
        # selected_arm_2 = choice(filtered_vals)

        # #filter out bad arms
        # if t < 500:
        #     if should_explore < 0.25:
        #         return randint(0, self.K-1), 1
        #     else:
        #         return selected_arm, 0
        # #run UCB
        # else:
        #     # return np.argmax(val), 1
        #     means = [val_ucb[index] for index in kuku_3]
        #     index_index = np.argmax(means)
        #     return kuku_3[index_index], 1
            

        # should_explore = np.random.rand()
        # if should_explore < 0.05:
        #     val = [arm.mean_reward() + np.sqrt(2*np.log(t)/arm.pulls) for (index, arm) in enumerate(self.arms)]
        #     return np.argmax(val), 1
        # else:
        #     return selected_arm, 0
            # crazy = np.zeros((self.K, 3))
            # crazy[:, 0] = [index for index in range(self.K)]
            # crazy[:, 1]  = np.array(q_tilde)

            # #expected mean
            # val = np.array([arm.mean_reward() for arm in self.arms])
            # crazy[:, 2] = val

            
        #     # crazy = sorted(crazy, key = operator.itemgetter(1, 2))

        #     # # print(crazy)
        #     # final = crazy[-1][0]
        #     # return int(final), True
        #     return np.argmax(q_tilde), 0.25
        # crazy = np.zeros((self.K, 3))
        # crazy[:, 0] = [index for index in range(self.K)]
        # crazy[:, 1]  = np.array(q_tilde)

        # #expected mean
        # val = np.array([arm.mean_reward() + np.sqrt(2*np.log(t)/arm.pulls) for arm in self.arms])
        # crazy[:, 2] = val

        # should_explore = np.random.rand()
        # if should_explore < 0.10:
        #     crazy = sorted(crazy, key = operator.itemgetter(2, 1))
        #     final = crazy[-1][0]
        #     return int(final), False
        # else:
        #     crazy = sorted(crazy, key = operator.itemgetter(1, 2))
        #     final = crazy[-1][0]
        #     return int(final), True

        

    def filter_arms(self, pulls, t):
        filtered_indices = []
        sorted_pulls = np.argsort(pulls)
        total_sum = np.sum(pulls)
        temp_sum = np.sum(pulls)
        # 1/np.log(.001*t+np.exp(1))
        for index in sorted_pulls:
            if (temp_sum - pulls[index])/total_sum >= 0.95:
                temp_sum -= pulls[index]
            else:
                filtered_indices.append(index)

        return filtered_indices

    def compute_fun(self, pulls, a= 1):
        total_sum = np.sum(pulls)
        diff_vector = total_sum - np.array(pulls)
        for index, val in enumerate(diff_vector):
            if pulls[index] >= 1 + a*val:
                return True

        return False
            

    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
