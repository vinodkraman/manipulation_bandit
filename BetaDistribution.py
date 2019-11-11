from scipy.stats import beta
import numpy as np
from Distribution import Distribution

class BetaDistribution(Distribution):
    def __init__(self, alpha = 1, beta = 1):
        self.alpha = alpha
        self.beta = beta 
    
    def mean(self):
        return (self.alpha)/ (self.alpha + self.beta)
    
    def update(self, delta_alpha, delta_beta):
        self.alpha += delta_alpha
        self.beta += delta_beta

    def sample(self):
        return np.asscalar(beta.rvs(self.alpha, self.beta, size=1))

    def get_params(self):
        return self.alpha, self.beta