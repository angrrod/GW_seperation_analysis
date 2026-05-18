from abc import ABC, abstractmethod
from scenario import GWScenario
from .Pipeline import Pipeline

class Pipeline_Amortized(Pipeline):
    def __init__(self,logger,scenario:GWScenario):
        super().__init__(logger, scenario)
        
    @abstractmethod
    def set_up(self):
        """
        load data and sets up config files to be translated from bilby/scenario to dingo
        """
        raise NotImplementedError
        
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def infer(self):
        raise NotImplementedError