import bilby
import copy
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from .SingleSignalMethod import SingleSignalMethod
from scenario import GWScenario
class HyrarchicalMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):
        super().__init__(run_sampler, scenario, logger, config,start_from_chekpt)
        self.method_type     = Method_type.HIERARCHICAL
        self.singleSampler   = SingleSignalMethod(True, scenario, logger,config) #run_sampler to true so that we always generate a new sample instead of using the one from the single method
        self.singleSampler.nameExtra = "1" 
        self.second_wave_ifos = self.scenario.ifos
        
    def generateSamples(self):
        self.logger.info("$$$ generate Samples for hyrarchical model")
        self.singleSampler.generateSamples()
        resultsSampleA       = copy.deepcopy(self.singleSampler.results["waveFormA"])
        MLPosteriorA         = self._getMaximumLikelihood(resultsSampleA)
        
        self.second_wave_ifos = self.getResidualIfos(MLPosteriorA)
        
        self.scenario.makePlots(["strain_time_domain_set_up_hyrarchical","qtransform_set_up_hyrarchical"],"",self.second_wave_ifos)
        super().generateSamples()
        resultsSampleB     = self.results['waveFormA'] #if we marginalize, the resampeling of the general posterior has been done via generateSamples() in the super class. 
        self.results = {
            "waveFormA" : resultsSampleA,
            "waveFormB" : resultsSampleB
        }
        
        
    def getLikelihood(self):
        self.logger.info("$$$ get likelihood sgnal for custom ifo")
        prior = self.prior
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers          = self.second_wave_ifos,
            waveform_generator       = self.scenario.wg,
            priors                   = prior,
            distance_marginalization = False,
            phase_marginalization    = False,
            time_marginalization     = False,
            # reference_frame="H1L1", #depends on the detector config -> ok?
            # time_reference="H1",
        )
        return likelihood
    
    def getPrior(self):
        return self.GetSinglePrior() 
