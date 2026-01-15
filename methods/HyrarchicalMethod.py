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
        self.method_type             = Method_type.HIERARCHICAL
        self.singleSampler           = SingleSignalMethod(True, scenario, logger,config,start_from_chekpt = False) #run_sampler to true so that we always generate a new sample instead of using the one from the single method
        self.singleSampler.nameExtra = "1" 
        self.second_wave_ifos        = self.scenario.ifos
        # (1) update the likelihood to have the second ifos
        self.likelihood              = self.getLikelihood() 
        
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
        if not hasattr(self, "second_wave_ifos"):
            return None # we will update the likelihood in this object not in the super class object. see (1)
        fiducial_parameters = self.scenario.injct_params_waves[0].copy()
        fiducial_parameters["time_jitter"] = 0.0
        likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(
            interferometers          = self.second_wave_ifos,
            waveform_generator       = self.scenario.wg_rel,
            priors                   = prior,
            fiducial_parameters      = fiducial_parameters,
            update_fiducial_parameters=True,
            distance_marginalization = False,
            phase_marginalization    = True,
            time_marginalization     = True,
            jitter_time              = False
        )
        return likelihood
    
    def getPrior(self):
        return self.GetSinglePrior() 
