import bilby
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario

class SingleSignalMethod(Method):
    def __init__(self, run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):
        super().__init__( run_sampler, scenario, logger, config)
        self.method_type = Method_type.SINGLE
        self.nameExtra = ""
    
    def sampeler(self,resume):
        #adapt the prior
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty for single signal")
        prior = self.GetSinglePrior()
        clean = not resume
        sample = bilby.run_sampler(
            likelihood = self.likelihood(),
            priors     = prior,
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
            outdir     = "out/outdir_ET_dynesty_" + self.method_type.code + self.nameExtra,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    def getPrior(self):
        return self.GetSinglePrior()
