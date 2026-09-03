import bilby
from .Method import Method
from scenario import GWScenario
from bilby.gw.conversion import generate_posterior_samples_from_marginalized_likelihood
class SingleLikelihoodMethod(Method):
    def __init__(self, scenario:GWScenario,logger,pipeline_type_code:str,nameExtra:str):
        super().__init__(scenario, logger, pipeline_type_code)
        self.nameExtra        = nameExtra #needed for getting the correct logging directories
        self._logl_diag_state = {"did_header": set()} # for logging purposes
        
    def getLikelihood_wrapped(self, ifos_override,newPriors:bool = False):
        #ovrride for logging/testing/debug reasons
        #code used in residual calculations
        ifos = self.parseIfos_override(ifos_override)
        
        self.logger.info("$$$ get single likelihood signal")
        if self.prior is None:
            raise NotImplementedError("prior is not implemented")
        
        #after marginalization the likelihood needs to be recreated for testing (in singleLikelihood pipeline)
        elif newPriors:
            prior = self.GetSinglePrior()
        else:
            prior = self.prior
        
        if self.configs["scenario"]["UseRelBinning"]:
            fiducial_parameters = self.scenario.injct_params_waves[0].copy()
            fiducial_parameters["time_jitter"] = 0.0
            likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(  #GravitationalWaveTransient
                interferometers            = ifos,
                waveform_generator         = self.wg_rel,
                priors                     = prior,
                fiducial_parameters        = fiducial_parameters,
                update_fiducial_parameters = False, #no optimization is done since we use the truth already
                distance_marginalization   = False,
                phase_marginalization      = True,
                time_marginalization       = True,
                jitter_time                = False
            )
        else:
            likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
                interferometers          = ifos,
                waveform_generator       = self.scenario.wg,  
                priors                   = self.prior,
                distance_marginalization = False,
                phase_marginalization    = True,
                time_marginalization     = True,
                jitter_time              = False,
            )
        return likelihood
    
    def getLikelihood(self,ifos_override,wrapped:bool = False,newPriors:bool = False):
        # wrapper object of the likelihood
        # build the real likelihood object from Method
        like = self.getLikelihood_wrapped(ifos_override,newPriors)
        # WJ: 13/02/2026 wrapped will always be false, it is old code that is used for logging
        # if wrapped:
        #     # attach diagnostic logging to this likelihood object
        #     theta_inj = self.scenario.injct_params_waves[0].copy()
        #     like = self.attach_logl_wrapper(
        #         like,
        #         theta_actual=theta_inj,
        #         every=1000,          # choose cadence
        #         label_actual="inj",
        #     )
        return like

    def updateResults(self,result,likelihood):
        # mutate the posterior by marginalizing it
        self.logger.info("$$$ marginalizing posterior samples")
        result.posterior = generate_posterior_samples_from_marginalized_likelihood(
            samples=result.posterior,      # what you read from HDF5
            likelihood=likelihood,     # rebuilt likelihood with marg flags enabled
            npool=18,                  # match your compute setting if you like
            block=50,
            use_cache=True,
        )
        return result
    
    def getPrior(self):
        self.logger.info("$$$ getting the prior")
        return self.priorConstructor.GetSinglePrior() 
    
    def log_diagnostic_tests(self,result,ifos_override):
        #tests for set-up function
        likelihood  = self.getLikelihood(ifos_override,newPriors = True)
        inj = self.scenario.injct_params_waves[0].copy()
        inj = {k: v for k, v in inj.items() if k in likelihood.priors}
        
        theta_inj = self.scenario.injct_params_waves[0].copy()
        # theta_ml  = self.getMaximumLikelihood(result)
        
        self._missing_dropped_keys_test(likelihood)
        self._bad_prior_support_test(likelihood,inj)
        self._probe_local_logl_test(likelihood,theta_inj)
        self._ML_inj_comparison_test(likelihood,inj,result)
