from bandits.bandit import Bandit
from distributions.betadistribution import BetaDistribution
import numpy as np
import copy
from scipy.stats import beta
from scipy.special import betainc
import matplotlib.pyplot as plt



x = np.linspace(0, 1.0, 100)
y1 = beta.pdf(x, 4, 6)
print(beta.ppf(0.50, 4, 6))
y2 = beta.pdf(x, 12, 18)
print(beta.ppf(0.50, 12, 18))
y3 = beta.pdf(x, 16, 24)
print(beta.ppf(0.50, 16, 24))
y4 = beta.pdf(x, 400, 600)
print(beta.ppf(0.50, 400, 600))
plt.plot(x, y1, label=1)
plt.plot(x, y2, label=2)
plt.plot(x, y3, label=3)
plt.plot(x, y4, label=4)
plt.legend()
plt.show()
