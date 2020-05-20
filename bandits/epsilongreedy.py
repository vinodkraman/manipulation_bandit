# from bandits.bandit import Bandit
# from random import *
# import numpy as np

# class EpsilonGreedy(Bandit):
#     def select_arm(self, t, q_tilde, W, C, order= None, influence_limit = False):
#         should_explore = random() < self.epsilon

#         # if should_explore:
#         #     return randint(0, self.K-1)
#         # else:
#         #     max_value = 0
#         #     selected_arm = 0
#         #     for (index, arm) in enumerate(self.arms):
#         #         val = arm.mean_reward()
#         #         if val > max_value:
#         #             max_value = val
#         #             selected_arm = index
#         #     return selected_arm

#         if t <= 1*len(self.arms):
#             return (t-1)%self.K, (t-1)%self.K

#         val_mean = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
#         val_arm_pulls = [arm.real_pulls for (index, arm) in enumerate(self.arms)]
#         filtered_vals = self.filter_arms(val_arm_pulls, t)

#         means = [val_mean[index] for index in filtered_vals]
#         index_index = np.argmax(means)
#         selected_arm = 0
#         if should_explore:
#             selected_arm = randint(0, self.K-1)
#         else:
#             selected_arm = filtered_vals[index_index]
        
#         return selected_arm, np.argmax(q_tilde)
        
#     def update_arm(self, arm, reward):
#         self.arms[arm].pulls += 1
#         self.arms[arm].rewards += reward

#     def filter_arms(self, pulls, t):
#         filtered_indices = []
#         sorted_pulls = np.argsort(pulls)
#         total_sum = np.sum(pulls)
#         temp_sum = np.sum(pulls)
#         # 1/np.log(.001*t+np.exp(1))
#         for index in sorted_pulls:
#             if (temp_sum - pulls[index])/total_sum >= 0.95:
#                 temp_sum -= pulls[index]
#             else:
#                 filtered_indices.append(index)

#         return filtered_indices
from bandits.bandit import Bandit
import numpy as np
from random import *
import operator

#the idea is to use expert advice to eliminate arms that we know are bad. Problem is, in the begininning, we cant immediatley follow expert advice, so we instead randomly selecet

class EpsilonGreedy(Bandit):

    def compute_KLD(self, p, q):
        if p == 0:
            return (1-p)*np.log((1-p)/(1-q))
        else: 
            # print(q)
            return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

    def select_arm(self, t, q_tilde, W, C, order= None): 
        should_explore = np.random.rand() < 0.20      
        if t <= 1*len(self.arms):
            return (t-1)%self.K, 0

        val_mean = [arm.mean_reward() for (index, arm) in enumerate(self.arms)]
        val_arm_pulls = [arm.real_pulls for (index, arm) in enumerate(self.arms)]
        filtered_vals = self.filter_arms(val_arm_pulls, t)
        # print(val_arm_pulls)
        # print(filtered_vals)


        means = [val_mean[index] for index in filtered_vals]
        index_index = np.argmax(means)
        selected_arm = 0

        should_kut = np.random.rand()
        if should_kut < 0.80:
            if should_explore:
                selected_arm = choice(filtered_vals)
            else:
                selected_arm = filtered_vals[index_index]
        
            return selected_arm, 1
        else:
            return np.argmax(q_tilde), -1


        # return filtered_vals[index_index], np.argmax(q_tilde)

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
            if (temp_sum - pulls[index])/total_sum >= 0.99:
                temp_sum -= pulls[index]
            else:
                filtered_indices.append(index)

        return filtered_indices

    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
