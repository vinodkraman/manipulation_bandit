from bandits.bandit import Bandit
import numpy as np

class UCB_influence(Bandit):

    def select_arm(self, t, q_tilde, W, C):
        selected_arm = 0
        max_value = 0
        # print(W)
        if t <= len(self.arms):
            return t-1
        else:
            #select arm
            for (index, arm) in enumerate(self.arms):
                # eta = min(1 - 1/np.exp(W), 1 - arm.pulls/(arm.pulls+C))

                # if arm.pulls > 100:
                #     eta = 1
                # else:
                eta = 1/np.exp(W)

                
                # eta = max((1/np.exp(W)), arm.pulls/(arm.pulls+50))

                #maximum possible value for eta to be capped so if the arm is pulled 50 times, then we want the max of eta to be 1/2 or lower
                # print(eta)
                # val = arm.mean_reward()*(arm.pulls)/(arm.pulls + W) + q_tilde[index] * (W/(arm.pulls + W)) + np.sqrt((2 * np.log(t)/((arm.pulls) + W)))
                # val = arm.mean_reward()*(1)/(np.exp(W)) + q_tilde[index] * (1 - 1/(np.exp(W))) + np.sqrt((2 * np.log(t)/((arm.pulls) + W)))
                # val = arm.mean_reward()*(1)/(np.exp(W)) + q_tilde[index] * (1 - 1/(np.exp(W)))
                val = (arm.mean_reward() + np.sqrt((2 * np.log(t)/arm.pulls)))*(eta) + q_tilde[index] * (1-eta)
                # val = (arm.mean_reward())*(eta) + q_tilde[index] * (1-eta)
                # val = arm.mean_reward() + np.sqrt((2 * np.log(t)/arm.pulls))

                if val > max_value:
                    max_value = val
                    selected_arm = index

            return selected_arm
        
    def update(self, arm, reward):
        self.arms[arm].pulls += 1
        self.arms[arm].rewards += reward
