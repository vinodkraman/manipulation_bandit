from abc import ABC, abstractmethod
from BanditArm import BanditArm
from BetaDistribution import BetaDistribution
import copy

class Bandit(ABC):
    def __init__(self, T, K, world_priors, epsilon = 0.8):
        self.K = K
        self.epsilon = epsilon
        self.world_priors = world_priors
        # self.arms = [BanditArm(copy.copy(prior)) for k in range(K)]
        self.arms = [BanditArm(copy.copy(prior)) for prior in world_priors]
        self.T = T
        super().__init__()
    
    @abstractmethod
    def select_arm(self):
        pass

    @abstractmethod 
    def update(self):
        pass

    def reset(self):
        self.arms = [BanditArm(copy.copy(prior)) for prior in self.world_priors]