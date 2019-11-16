from abc import ABC, abstractmethod
from BanditArm import BanditArm
from BetaDistribution import BetaDistribution
import copy

class Bandit(ABC):
    def __init__(self, K = 10, epsilon = 0.8, prior = BetaDistribution()):
        self.K = K
        self.epsilon = epsilon
        self.arms = [BanditArm(copy.copy(prior)) for k in range(K)]
        super().__init__()
    
    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod 
    def update_arm(self):
        pass