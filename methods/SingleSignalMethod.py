import bilby
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario
from bilby.core.sampler.dynesty import Dynesty, dynesty_stats_plot
import os
import shutil
class SingleSignalMethod(Method):
    def __init__(self, run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):
        super().__init__( run_sampler, scenario, logger, config)
        self.method_type = Method_type.SINGLE
        self.nameExtra = ""
    
    def sampeler(self,resume: bool):
        #adapt the prior
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty for single signal")
        prior = self.GetSinglePrior()
        outdir = "logs/log_ET_dynesty_" + self.method_type.code + self.nameExtra
        if not resume and os.path.isdir(outdir):
            self.logger.warning(f"$$$ Removing existing outdir for fresh run: {outdir}")
            shutil.rmtree(outdir)
        sampler = Dynesty(
            likelihood = self.likelihood(),
            priors     = prior,
            nlive      = self.config.nlive, 
            dlogz      = self.config.dlogz, #stopping criterion for the evidence
            sample     = self.config.sample,  
            walks      = self.config.walks, #steps for MCMC sampeler to select new candidates  
            bound      = self.config.bound,
            maxmcmc    = self.config.maxmcmc,
            nact       = self.config.nact, #amount of steps is tuned so autocorr is small enough 
            resume     = resume,
            outdir     = outdir,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        result = sampler.run_sampler()
        
        #store diagnostics plot
        fig, _   = dynesty_stats_plot(sampler)
        fileName = "sampler_diagnostics"
        path     = os.path.join(self.diagOutDir, f"{fileName}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        return result
    
    def log_inj_likel(self,result):
        #debug funciton
        #TODO: remove
        likelihood  = self.likelihood()

        inj = self.scenario.injct_params_waves[0].copy()
        inj = {k: v for k, v in inj.items() if k in likelihood.priors}

        likelihood.parameters.update(inj)
        logL_inj   = likelihood.log_likelihood()
        logLR_inj  = likelihood.log_likelihood_ratio()

        # ML point from the result
        ml = self._getMaximumLikelihood(result)   # your helper
        likelihood.parameters.update(ml)
        logL_ml    = likelihood.log_likelihood()
        logLR_ml   = likelihood.log_likelihood_ratio()

        self.logger.info(f"logL(inj)  = {logL_inj:.3f}")
        self.logger.info(f"logL(ml)   = {logL_ml:.3f}")
        self.logger.info(f"ΔlogL      = {(logL_ml-logL_inj):.3f}")

        self.logger.info(f"logLR(inj) = {logLR_inj:.3f}")
        self.logger.info(f"logLR(ml)  = {logLR_ml:.3f}")
        self.logger.info(f"ΔlogLR     = {(logLR_ml-logLR_inj):.3f}")
        
        logLR_max = float(result.log_likelihood_evaluations.max())
        logL_noise = float(result.log_noise_evidence)   # this is log L_noise
        logL_full = logLR_max + logL_noise
        
        self.logger.info(f"$$$ logLR_max = {logLR_max}, logL_noise = {logL_noise}, logL_full = {logL_full}")
    def getPrior(self):
        return self.GetSinglePrior()
