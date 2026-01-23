from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
from methods import MethodConfig, RunMode, JointLikelihoodlMethod

class JointLikelihoodPipeline(Pipeline):
    def __init__(self, logger, scenario:GWScenario, config:MethodConfig):
        super().__init__(logger, scenario, config)
        self.pipeline_type = Pipeline_type.JOINT
        self.jointLikl     = JointLikelihoodlMethod(scenario, logger, config,self.pipeline_type.code,)
    
    def run(self,runMode:RunMode):
        self.logger.info("$$$ Run the Joint Likelihood pipeline")
        resultA,resultB = self.jointLikl.run(runMode,ifos_override = None)
        results         = {
            "waveFormA" : resultA,
            "waveFormB" : resultB
            }
        return results
