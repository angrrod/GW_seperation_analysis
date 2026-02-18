from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from methods import DynestyConfig, RunMode, JointLikelihoodlMethod

class JointLikelihoodPipeline(Pipeline):
    def __init__(self, logger, scenario:GWScenario):
        super().__init__(logger, scenario)
        self.pipeline_type = Pipeline_type.JOINT
        self.jointLikl     = JointLikelihoodlMethod(scenario, logger, self.config, self.pipeline_type.code)
    
    def run(self,runMode:RunMode):
        self.logger.info("$$$ Run the Joint Likelihood pipeline")
        resultA,resultB = self.jointLikl.run(runMode,ifos_override = None)
        results         = {
            "waveFormA" : resultA,
            "waveFormB" : resultB
            }
        return results
    
    def getConfig(self):
        return DynestyConfig()
    
    def log_diagnostic_tests(self,result,ifos_override = None):
        # run diagnostic tests
        self.jointLikl.log_diagnostic_tests(result,ifos_override)
