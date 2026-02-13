from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from methods import PymcNutsConfig,SingleLikelihoodMethod, RunMode

class MetroWithinGibbsPipeline(Pipeline):
    #TODO: implement cosine, sine and geomcentric distance priors
    #      make sure arguents in single likelihood are passed correctly
    def __init__(self, logger, scenario:GWScenario):
        super().__init__(logger, scenario)
        # self.pipeline_type = TODO
        self.singleLikl    = SingleLikelihoodMethod(scenario, logger, self.config,self.pipeline_type.code, "")
    
    def run(self,runMode:RunMode):
        raise NotImplementedError
    
    def getConfig(self):
        return PymcNutsConfig()
    
    