from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from methods import SingleLikelihoodMethod, RunMode
from config import DynestyConfig,PymcNutsConfig

class SingleLikelihoodPipeline(Pipeline):
    def __init__(self, logger, scenario:GWScenario):
        super().__init__(logger, scenario)
        self.pipeline_type = Pipeline_type.SINGLE
        extraName          = "_temp_NUTS" #for logging purposes
        self.singleLikl    = SingleLikelihoodMethod(scenario, logger, self.config,self.pipeline_type.code, extraName)
    
    def run(self,runMode:RunMode):
        self.logger.info("$$$ Run the Single Likelihood pipeline")
        result  = self.singleLikl.run(runMode,ifos_override = None)
        results = {"waveFormA" : result}
        return results
    
    def log_diagnostic_tests(self,result,ifos_override = None):
        # run diagnostic tests
        self.singleLikl.log_diagnostic_tests(result,ifos_override)
        
    def getConfig(self):
        return DynestyConfig() #PymcNutsConfig() #DynestyConfig()
