import bilby
from bilby.gw.detector import get_empty_interferometer, InterferometerList
import copy
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from .SingleSignalMethod import SingleSignalMethod
from scenario import GWScenario

class HyrarchicalMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):
        super().__init__(run_sampler, scenario, logger, config)
        self.method_type     = Method_type.HIERARCHICAL
        self.singleSampler   = SingleSignalMethod(True, scenario, logger,config) #run_sampler to true so that we always generate a new sample instead of using the one from the single method
        self.singleSampler.nameExtra = "1" 
        self.second_wave_ifos = self.scenario.ifos
        
    def generateSamples(self):
        self.logger.info("$$$ generate Samples for hyrarchical model")
        self.singleSampler.generateSamples()
        posteriorSampleA     = copy.deepcopy(self.singleSampler.posteriors["waveFormA"])
        MLPosteriorA         = self._getMaximumLikelihood(posteriorSampleA)
        pols                 = self.scenario.wg.frequency_domain_strain(MLPosteriorA) #returns cross and plus waveform
        second_wave_ifos     = []
        for ifo in self.scenario.ifos:
            h_fd                 = ifo.get_detector_response(pols, MLPosteriorA)
            d_fd                 = ifo.strain_data.frequency_domain_strain
            res_fd               = d_fd - h_fd
            second_wave_ifo      = self._GetIfoResidual(res_fd,ifo)
            second_wave_ifos.append(second_wave_ifo)
        self.second_wave_ifos = InterferometerList(second_wave_ifos)
        self.scenario.makePlots(["strain_time_domain_set_up_hyrarchical","qtransform_set_up_hyrarchical"],self.second_wave_ifos)
        super().generateSamples()
        posteriorSampleB     = self.posteriors['waveFormA']
        self.posteriors = {
            "waveFormA" : posteriorSampleA,
            "waveFormB" : posteriorSampleB
        }
        
        
    def likelihood(self):
        self.logger.info("$$$ get likelihood sgnal for custom ifo")
        prior = self.GetSinglePrior()
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

    def _GetIfoResidual(self,res_fd,ifo):
        self.logger.info("$$$ get residual interferrometer")
        #build copy for second interferrometer
        new_ifo = get_empty_interferometer(ifo.name)
        new_ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain = res_fd,
            sampling_frequency      = ifo.strain_data.sampling_frequency,
            duration                = ifo.strain_data.duration,
            start_time              = ifo.strain_data.start_time,
        )
        new_ifo.power_spectral_density = ifo.power_spectral_density
        return new_ifo
    
    def getPrior(self):
        return self.GetSinglePrior()