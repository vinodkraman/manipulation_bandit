from BayesianGreedy import BayesianGreedy
from BanditArm import BanditArm
from BetaDistribution import BetaDistribution

# class BayesianGreedy(Bandit):
#     def select_arm(self, t, influence_limit = False):
#         should_explore = random() > self.epsilon
#         if should_explore:
#             return randint(0, self.K-1)
#         else:
#             max_value = 0
#             selected_arm = 0
#             for (index, arm) in self.arms.items():
#                 val = arm.reward_dist_mean(influence_limit = influence_limit)
#                 if val > max_value:
#                     max_value = val
#                     selected_arm = index
#             return selected_arm
        
#     def update_arm(self, arm, reward):
#         self.arms[arm].pulls += 1
#         self.arms[arm].rewards += reward
#         self.arms[arm].update_reward_dist(reward)

#Test:
prior = BetaDistribution(4, 5)
bayesian_greedy = BayesianGreedy(K = 3, epsilon=1, prior= prior)

bayesian_greedy.update_arm(0, 0)
bayesian_greedy.update_arm(1, 1)
bayesian_greedy.update_arm(1, 1)
bayesian_greedy.update_arm(2, 1)
arm_means = [4/10, 6/11, 5/10]
arm_pulls = [1, 2, 1]
arm_rewards = [0, 2, 1]
reward_means = [0, 1, 1]

#check update
for (index, arm) in bayesian_greedy.arms.items():
    assert arm.pulls == arm_pulls[index]
    assert arm.rewards == arm_rewards[index]
    assert arm.mean_reward() == reward_means[index] 
    assert arm.reward_dist_mean() == arm_means[index]

#check select_arm
selected_arm = bayesian_greedy.select_arm(0)
assert(selected_arm == 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
bayesian_greedy.update_arm(0, 1)
selected_arm = bayesian_greedy.select_arm(0)
assert(selected_arm == 0)


bayesian_greedy.update_arm(0, 1, True)
bayesian_greedy.update_arm(0, 1, True)
bayesian_greedy.update_arm(1, 0, True)
bayesian_greedy.update_arm(2, 1, True)
influence_arm_means = [6/11,4/10, 5/10]

#check update
for (index, arm) in bayesian_greedy.arms.items():
    assert arm.reward_dist_mean(influence_limit=True) == influence_arm_means[index]

#check select_arm
selected_arm = bayesian_greedy.select_arm(0, influence_limit=True)
assert(selected_arm == 0)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
bayesian_greedy.update_arm(1, 1, influence_limit=True)
selected_arm = bayesian_greedy.select_arm(0, influence_limit=True)
assert(selected_arm == 1)