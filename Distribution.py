from abc import ABC, abstractmethod
class Distribution(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def mean(self):
        pass

    @abstractmethod 
    def update(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_params(self):
        pass