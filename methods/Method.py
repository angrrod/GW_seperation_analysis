from abc import ABC, abstractmethod
import bilby
from bilby.core.result import read_in_result
import copy
from scenario import GWScenario
from .MethodConfig import MethodConfig

class Method(ABC):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):

        self.run_sampler = run_sampler
        self.scenario    = scenario
        self.logger      = logger
        self.method_type = None
        self.config      = config
        
        #results
        self.posteriors = None
    
    def likelihood(self):
        self.logger.info("$$$ get a single likelihood signal")
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers          = self.scenario.ifos,
            waveform_generator       = self.scenario.wg,
            priors                   = self.getPrior(),
            distance_marginalization = False,
            phase_marginalization    = False,
            time_marginalization     = False,
            # reference_frame="H1L1", #depends on the detector config -> ok?
            # time_reference="H1",
        )
        return likelihood
    
    def sampeler(self,resume):
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty")
        clean = not resume
        sample = bilby.run_sampler(
            likelihood = self.likelihood(),
            priors     = self.getPrior(),
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
            outdir     = "out/outdir_ET_dynesty_" + self.method_type.code,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    def generateSamples(self):
        self.logger.info("$$$ run the samples")
        #bayesian part
        if self.run_sampler:
            full_rerun = not self.run_sampler
            result = self.sampeler(full_rerun)
        else:
            outdir = "out/outdir_ET_dynesty_" + self.method_type.code + "/" + self.method_type.code + "_result.json"
            result = read_in_result(outdir) #outdir is also used in sampeler 
        self.posteriors = {"waveFormA" : result} #pandas data frame of samples
        
    ###   priors   ###
    def GetSinglePrior(self):
        self.logger.info("$$$ getting a waveform prior")
        prior = bilby.gw.prior.BBHPriorDict()  #allow for default ranges in ET
        if "geocent_time" not in prior:
            self.logger.info("$$$ geocent_time not in prior")
            prior["geocent_time"] = bilby.core.prior.Uniform(
                minimum=0,#maybe make this a bit bigger?
                maximum=self.scenario.config.duration,  
                name="geocent_time",
            )
        return prior
    
    #needed for joint parameter estimation
    # independent priors for both
    def getJointPriors(self):
        self.logger.info("$$$ getting joint priors")
        base = self.GetSinglePrior()   # BBHPriorDict
        priors = bilby.core.prior.PriorDict()

        for key, prior in base.items():
            priors[f"{key}_A"] = copy.deepcopy(prior)
            priors[f"{key}_B"] = copy.deepcopy(prior)

        return priors
    
    @abstractmethod
    def getPrior(self):
        pass

    def _getMaximumLikelihood(self,result):
        posterior = result.posterior
        idx_ml    = posterior["log_likelihood"].idxmax()
        ml_sample = posterior.loc[idx_ml]
        return {k: ml_sample[k] for k in result.search_parameter_keys} #format for waveform generator
    


    

    

