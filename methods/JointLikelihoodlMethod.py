import bilby
from jointRB.JointLikelihoodRB import OverlappingSignalsRelBinning
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario

class JointLikelihoodlMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):
        super().__init__(run_sampler, scenario, logger, config,start_from_chekpt)
        self.method_type = Method_type.JOINT
        
    def likelihood(self):
        #adapt both prior and likelihood for joint modelling
        self.logger.info("$$$ get the joint likelihood signal")
        waveform_parms = self.scenario.GetWaveFormParams()
        ref_injection  = self.scenario.build_ref_injection(waveform_parms)
        
        wg_rb          = self._getRBWaveForm()
        likelihood     = OverlappingSignalsRelBinning(
            interferometers    = self.scenario.ifos,
            waveform_generator = wg_rb,
            ref_injection      = ref_injection, # actual parameters in simulation, ML for actual data, this is the FUDICIAL waveform used in the RB scheme
            N_overlaps         = 2,
            priors             = self.getPrior(),
            reference_frame    = "sky",
            time_reference     = "geocenter",
            delta              = 0.03,  # RB binning tolerance
        )
        return likelihood
    
    def _getRBWaveForm(self):
        """
            wavform generator tailored for jointRB 
        """
        self.logger.info("$$$ get relative binning waveform generator")
        wg_rb = bilby.gw.waveform_generator.WaveformGenerator(
            duration                      = self.scenario.wg.duration,
            sampling_frequency            = self.scenario.wg.sampling_frequency,
            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,  #jrb_lal_binary_black_hole self.scenario.wg.frequency_domain_source_model
            waveform_arguments            = self.scenario.wg.waveform_arguments.copy()
        )
        return wg_rb
    
    def getPrior(self):
        return self.getJointPriors()