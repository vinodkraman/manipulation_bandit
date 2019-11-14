from scipy.stats import bernoulli
import numpy as np
from Distribution import Distribution

class BernoulliDistribution(Distribution):
    def __init__(self, theta):
        self.theta = theta
    
    def mean(self):
        return self.theta
    
    def update(self):
        pass

    def sample(self):
        return np.asscalar(bernoulli.rvs(self.theta, size=1))

    def get_params(self):
        return self.theta
    
    def set_params(self, theta):
        self.theta = theta