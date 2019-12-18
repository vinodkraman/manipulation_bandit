from scipy.stats import beta
import numpy as np
from Distribution import Distribution

class BetaDistribution(Distribution):
    def __init__(self, alpha= 1, beta= 1):
        self.alpha = alpha
        self.beta = beta 
    
    def mean(self):
        return (self.alpha)/ (self.alpha + self.beta)
    
    def update(self, data):
        self.alpha += (data == 1)
        self.beta += (data == 0)

    def sample(self):
        return np.asscalar(beta.rvs(self.alpha, self.beta, size=1))

    def get_params(self):
        return self.alpha, self.beta
    
    def set_params(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def get_quantile(self, prob):
        return beta.ppf(prob, self.alpha, self.beta)