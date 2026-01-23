from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from methods import MethodConfig, SingleLikelihoodMethod, RunMode

class SingleLikelihoodPipeline(Pipeline):
    def __init__(self, logger, scenario:GWScenario, config:MethodConfig):
        super().__init__(logger, scenario, config)
        self.pipeline_type = Pipeline_type.SINGLE
        extraName          = "" #for logging purposes
        self.singleLikl    = SingleLikelihoodMethod(scenario, logger, config,self.pipeline_type.code, extraName)
    
    def run(self,runMode:RunMode):
        self.logger.info("$$$ Run the Single Likelihood pipeline")
        result = self.singleLikl.run(runMode,ifos_override = None)
        results = {"waveFormA" : result}
        return results
    
    def log_diagnostic_tests(self,dataPipeline,ifos_override = None):
        # run diagnostic tests
        self.singleLikl.log_diagnostic_tests(dataPipeline,ifos_override)
