from BanditArm import BanditArm
from BetaDistribution import BetaDistribution

#Test:
reward_dist = BetaDistribution(2, 3)
influence_dist = BetaDistribution(2, 6)
bandit_arm = BanditArm(reward_dist)
bandit_arm.pulls = 5
bandit_arm.rewards = 2
bandit_arm.influence_reward_dist = influence_dist

assert (bandit_arm.mean_reward() == 2/5)

num_samples = 10000
total = 0
for i in range(num_samples):
    total += bandit_arm.sample()
assert(2/5 - 0.03 <= total/num_samples <= 2/5 + 0.03)

total = 0
for i in range(num_samples):
    total += bandit_arm.sample(influence_limit= True)
assert(2/8 - 0.03 <= total/num_samples <= 2/8 + 0.03)

assert(bandit_arm.reward_dist_mean() == 2/5)
assert(bandit_arm.reward_dist_mean(influence_limit=True) == 2/8)

bandit_arm.update_reward_dist(1)
bandit_arm.update_reward_dist(0)
bandit_arm.update_reward_dist(0)
bandit_arm.update_reward_dist(1)
assert(bandit_arm.reward_dist_mean() == 4/9)

bandit_arm.update_reward_dist(1, influence_limit=True)
bandit_arm.update_reward_dist(0, influence_limit=True)
bandit_arm.update_reward_dist(0, influence_limit=True)
bandit_arm.update_reward_dist(1, influence_limit=True)
assert(bandit_arm.reward_dist_mean(influence_limit=True) == 4/12)