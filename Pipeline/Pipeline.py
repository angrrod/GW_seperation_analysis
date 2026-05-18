
from abc import ABC, abstractmethod
from scenario import GWScenario

#general object for pipelines
class Pipeline(ABC):
    def __init__(self,logger,scenario:GWScenario):
        self.logger        = logger
        self.scenario      = scenario
        self.pipeline_type = None
        
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError