from methods import RunMode
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from .TasNetPipeline import TasNetPipeline
from .Pipeline_Sampler import Pipeline_Sampler
from config import DynestyConfig

class VectorCopulaPipeline(Pipeline_Sampler):
    def __init__(self,logger,scenario:GWScenario):
        super().__init__(logger, scenario)
        # self.pipeline_type  = Pipeline_type.VECTORCOPULA
        self.prePipeline = TasNetPipeline(logger,scenario) 
    
    def run(self,runMode:RunMode):
        results = self.prePipeline.run(runMode)
        resultsSampleA = results["waveFormA"]
        resultsSampleB = results["waveFormB"]
        

    def getConfig(self):
        return DynestyConfig()