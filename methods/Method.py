from abc import ABC, abstractmethod
import bilby
from bilby.core.result import read_in_result
import copy
from scenario import GWScenario
from .MethodConfig import MethodConfig
from bilby.core.prior import DeltaFunction
from bilby.core.result import read_in_result
from enum import Enum
from bilby.gw.detector import InterferometerList
class RunMode(Enum):
    RUN    = "run"
    REUSE  = "reuse"
    CHEKPT = "checkpoint"
    
class Method(ABC):
    def __init__(self,scenario:GWScenario,logger,config:MethodConfig,pipeline_type_code:str):
        self.scenario           = scenario
        self.logger             = logger
        self.config             = config
        self.prior              = self.getPrior()
        self.pipeline_type_code = pipeline_type_code
        self.wg_rel             = self._getRBWaveForm()
    
    def sample(self,resume,clean,ifos_override):
        #adapt the prior
        self.logger.info("$$$ Starting sampler")
        
        if self.prior is None:
            raise NotImplementedError("prior is not implemented")
        likelihood = self.getLikelihood(ifos_override)
        # TODO: DELETE
        # seed_theta = self.scenario.injct_params_waves[0]   # injection dict
        # live_points = self._make_seeded_live_points_dynesty(
        #     likelihood=likelihood,
        #     priors=self.prior,
        #     seed_theta=seed_theta,
        #     nlive=self.config.nlive,
        #     seed_frac=0.05,        # 10% of live points near injection
        #     rel_jitter=1e-3        # jitter scale relative to prior width
        # )
        
        sample = bilby.run_sampler(
            likelihood = likelihood,
            priors     = self.prior,
            sampler    = self.config.sampler,
            nlive      = self.config.nlive, 
            dlogz      = self.config.dlogz, #stopping criterion for the evidence
            sample     = self.config.sample,  
            walks      = self.config.walks, #steps for MCMC sampeler to select new candidates  
            bound      = self.config.bound,
            maxmcmc    = self.config.maxmcmc,
            nact       = self.config.nact, #amount of steps is tuned so autocorr is small enough 
            resume     = resume,
            clean      = clean,
            # live_points= live_points, #TODO: DELETE
            outdir     = "logs/log_ET_dynesty_" + self.pipeline_type_code + self.nameExtra,
            label      = self.pipeline_type_code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        results = self.updateResults(sample,likelihood)
        return results
    
    def run(self,runMode:RunMode,ifos_override):
        #main entry point
        self.logger.info("$$$ Generating posterior samples")
        if runMode == RunMode.REUSE:
            outdir = "logs/log_ET_dynesty_" + self.pipeline_type_code + "/" + self.pipeline_type_code + "_result.json"
            result = read_in_result(outdir) #outdir is also used in sampeler
            result = self.postprocessReusedResult(result)
            
        else: 
            if runMode == RunMode.RUN:
                clean  = True
                resume = False
            elif runMode == RunMode.CHEKPT:
                clean  = False
                resume = True
            result = self.sample(resume,clean,ifos_override)
        return result
    
    def parseIfos_override(self,ifos_override):
        if ifos_override is None:
            ifos = self.scenario.ifos
        else:
            # Ensure bilby receives an InterferometerList (works with list slices too)
            if isinstance(ifos_override, InterferometerList):
                ifos = ifos_override
            else:
                ifos = InterferometerList(ifos_override)
        return ifos
    
    @abstractmethod
    def updateResults(self,result,likelihood):
        # method used for postprocessing the result
        raise NotImplementedError
    
    @abstractmethod
    def getLikelihood(self,ifos_override):
        raise NotImplementedError
    
    def postprocessReusedResult(self, result):
        # default Nothing happens
        return result


##################
###   priors   ###
##################
    
    def GetSinglePrior(self,waveformIdx = 0):
        self.logger.info("$$$ getting a waveform prior")
        prior = bilby.gw.prior.BBHPriorDict()  #allow for default ranges in ET
        if "geocent_time" not in prior:
            self.logger.info("$$$ geocent_time not in default prior, adding manually")
            prior["geocent_time"] = bilby.core.prior.Uniform(
                minimum=0,#maybe make this a bit bigger?
                maximum=self.scenario.config.duration,  
                name="geocent_time",
            )
        prior["chirp_mass"] = bilby.core.prior.Uniform(
            minimum=4, maximum=50, name="chirp_mass"
        )
        prior["mass_ratio"] = bilby.core.prior.Uniform(
            minimum=0.1, maximum=1, name="mass_ratio" 
        )
        prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=1e3,      
            maximum=1e5, #1e5      
            cosmology='Planck15',
            name='luminosity_distance',
            latex_label='$d_L$',
            unit='Mpc'
        )
        
        ### WJ: 04/01/25 fix priors for debugging
        # fixed_priors   = ["tilt_1", "tilt_2", "phi_12", "phi_jl", "a_1", "a_2"]
        # waveFormParams = self.scenario.GetWaveFormParams()
        # for k in fixed_priors:
        #     if k in prior:
        #         self.logger.info(f"$$$ : making prior delta {k}")
        #         value = waveFormParams[waveformIdx].get(k)
        #         prior[k] = DeltaFunction(value, name=k)
        
        return prior
    
    # needed for joint parameter estimation
    # independent priors for both
    def getJointPriors(self):
        self.logger.info("$$$ getting joint priors")
        base = self.GetSinglePrior()   # BBHPriorDict
        priors = bilby.core.prior.PriorDict()

        for key, prior in base.items():
            priors[f"{key}_A"] = copy.deepcopy(prior)
            priors[f"{key}_B"] = copy.deepcopy(prior)

        return priors
    
    def _getRBWaveForm(self):
        """
            wavform generator tailored for jointRB 
        """
        self.logger.info("$$$ get relative binning waveform generator")
        wg_rb = bilby.gw.waveform_generator.WaveformGenerator(
            duration                      = self.scenario.wg.duration,
            sampling_frequency            = self.scenario.wg.sampling_frequency,
            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole_relative_binning, #lal_binary_black_hole  #jrb_lal_binary_black_hole self.scenario.wg.frequency_domain_source_model
            parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments            = self.scenario.wg.waveform_arguments.copy()
        )
        return wg_rb   
    
    @abstractmethod
    def getPrior(self):
        pass
