import bilby
from jointRB import OverlappingSignalsRelBinning
from .Method import Method
from scenario import GWScenario
from prior import prior
from config import ScenarioConfig

class JointLikelihoodlMethod(Method):
    def __init__(self,scenario:GWScenario,logger,config,pipeline_type_code:str):
        super().__init__(scenario, logger, config, pipeline_type_code)
        self.nameExtra = ""
        
    def getLikelihood(self,ifos_override):
        #adapt both prior and likelihood for joint modelling
        self.logger.info("$$$ get the joint likelihood signal")
        waveform_parms = self.scenario.injct_params_waves
        ref_injection  = self.scenario.build_ref_injection(waveform_parms)
        
        #redundant code used to unsure uniformity with SingleLikelihood
        ifos = self.parseIfos_override(ifos_override)
                
        if self.prior is None:
            raise NotImplementedError("prior is not implemented")
        
        if not self.scenario.config.UseRelBinning:
            raise NotImplementedError("joint likelihood only implemented with Relative binning")
        if self.scenario.config.UseRelBinning:
            waveform_generator = self.wg_rel
        else:
            waveform_generator = self.scenario.wg
        
        # waveform generator for relative binning likelihood model
        likelihood = OverlappingSignalsRelBinning(
            interferometers    = ifos,
            waveform_generator = waveform_generator,
            ref_injection      = ref_injection, # actual parameters in simulation, ML for actual data, this is the FUDICIAL waveform used in the RB scheme
            N_overlaps         = 2,
            priors             = self.prior,
            reference_frame    = "sky",
            time_reference     = "geocenter",
            delta              = 0.001,  # RB binning tolerance
        )
        
        _base_log_likelihood_ratio = likelihood.log_likelihood_ratio
        def wrapped_log_likelihood_ratio(parameters=None):
            # wrapped likelihood, add constraint on the prior to force the likelihood
            p = parameters if parameters is not None else likelihood.parameters
            p = dict(p)
            if "geocent_time_A" in p and "delta_t_AB" in p:
                p["geocent_time_B"] = p["geocent_time_A"] - p["delta_t_AB"]
            elif "geocent_time_B" in p and "delta_t_AB" in p:
                p["geocent_time_A"] = p["geocent_time_B"] + p["delta_t_AB"]
            else:
                raise KeyError(
                    "Need either (geocent_time_A, delta_t_AB) or "
                    "(geocent_time_B, delta_t_AB) to reconstruct the joint times."
                )

            likelihood.parameters.update(p)
            return _base_log_likelihood_ratio()

        
        likelihood.log_likelihood_ratio = wrapped_log_likelihood_ratio
        return likelihood
        
    def updateResults(self,result,likelihood = None):
        self.logger.info("$$$ split waveforms for result object")
        posterior    = result.posterior
        posterior["geocent_time_B"] = (
            posterior["geocent_time_A"]
            - posterior["delta_t_AB"]
        )
        resultA      = posterior.loc[:, posterior.columns.str.endswith("_A")].copy()
        resultB      = posterior.loc[:, posterior.columns.str.endswith("_B")].copy()
        resultA      = resultA.rename(columns=lambda c: c[:-2])
        resultB      = resultB.rename(columns=lambda c: c[:-2])
        return resultA,resultB
    
    def postprocessReusedResult(self, result):
        #split the processor
        return self.updateResults(result)

    def getPrior(self):
        self.logger.info("$$$ getting the prior")
        return self.priorConstructor.getJointPriors()
    
    def log_diagnostic_tests(self,result,ifos_override):
        #tests for set-up function
        theta_inj = self.scenario.injct_params_waves.copy()
        # theta_ml  = self.getMaximumLikelihood(result)
        ref_injection  = self.scenario.build_ref_injection(theta_inj.copy())
        
        likelihood  = self.getLikelihood(ifos_override)
        inj = {k: v for k, v in ref_injection.copy().items() if k in likelihood.priors}
        
        # theta_ml  = self.getMaximumLikelihood(result)
        
        self._missing_dropped_keys_test(likelihood)
        self._bad_prior_support_test(likelihood,inj)
        self._probe_local_logl_test(likelihood,ref_injection)
        # self._ML_inj_comparison_test(likelihood,inj,result)