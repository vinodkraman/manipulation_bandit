from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
from distributions.bernoullidistribution import BernoulliDistribution
from scipy.stats import bernoulli
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
import matplotlib.pyplot as plt



x = np.linspace(0, 1.0, 100)
y2 = beta.pdf(x, 2, 3)
alpha = 1
bet = 1



for t in range(5):
    sample = np.asscalar(beta.rvs(2, 3, size=1))
    bern = BernoulliDistribution(sample)
    val = bern.sample()

    if val == 1:
        alpha += 1
    else:
        bet += 1

y_test = beta.pdf(x, alpha, bet)

plt.plot(x, y2, label= 1)
plt.plot(x, y_test, label= 2)
plt.legend()
plt.show()

