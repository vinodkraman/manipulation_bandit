from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
from distributions.bernoullidistribution import BernoulliDistribution
from scipy.stats import bernoulli
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
# fig, ax = plt.subplots(1, 1)

# # x = np.linspace(0, 1.0, 100)
# # y2 = beta.pdf(x, 2, 3)
# # alpha = 1
# # bet = 1

# a = 0.12 * 100 + .30 * 100 + .23 * 92 + .24 * 100 + .11 * 100
# print(a) 

# for t in range(5):
#     sample = np.asscalar(beta.rvs(2, 3, size=1))
#     bern = BernoulliDistribution(sample)
#     val = bern.sample()

#     if val == 1:
#         alpha += 1
#     else:
#         bet += 1

# y_test = beta.pdf(x, alpha, bet)

# plt.plot(x, y2, label= 1)
# plt.plot(x, y_test, label= 2)
# plt.legend()
# plt.show()

# # s = np.random.normal(0, 0.15, 10)
# # print(s)


# # a = (-1*0.3)/0.10 
# # b = (1-0.3)/0.10
# # mean, var, skew, kurt = truncnorm.stats(a, b, moments='mvsk')


# # x = np.linspace(truncnorm.ppf(0.01, a, b),
# #                 truncnorm.ppf(0.99, a, b), 100)
# # ax.plot(x, truncnorm.pdf(x, a, b, 0.3, 0.10),
# #        'r-', lw=5, alpha=0.6, label='truncnorm pdf')


# # rv = truncnorm(a, b)
# # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')


# # vals = truncnorm.ppf([0.001, 0.5, 0.999], a, b)
# # np.allclose([0.001, 0.5, 0.999], truncnorm.cdf(vals, a, b))



# # r = truncnorm.rvs(a, b, size=1000)


# # ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
# # ax.legend(loc='best', frameon=False)
# plt.show()



# #so the issue is, if the experts do not perform the sleeper attack, then we are fine, since we can just greedy exploit q_tilde. If the experts 
# #exploit the sleeper attack, then there is no way for us to be able to do anything. 

s = np.random.binomial(10000, 0.6)/10000
print(s)
# s = np.random.binomial(5, 0.75, 1)
