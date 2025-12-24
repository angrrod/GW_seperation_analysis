import bilby
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario
import numpy as np
class SingleSignalMethod(Method):
    def __init__(self, run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):
        super().__init__( run_sampler, scenario, logger, config,start_from_chekpt)
        self.method_type = Method_type.SINGLE
        self.nameExtra = ""
    
    def sampeler(self):
        #adapt the prior
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty for single signal")
        prior = self.GetSinglePrior()       
        if self.run_sampler:
            clean = not self.start_from_chekpt  #do we need to clean the code
        else:
            clean = False
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
            resume     = not self.run_sampler,
            clean      = clean,
            outdir     = "logs/log_ET_dynesty_" + self.method_type.code + self.nameExtra,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    # diagnostic tests
    def _missing_dropped_keys_test(self,likelihood):
        # dropped or missing keys?
        inj_full = self.scenario.injct_params_waves[0].copy()
        prior_keys = set(likelihood.priors.keys())
        inj_keys = set(inj_full.keys())

        dropped = sorted(inj_keys - prior_keys)
        missing = sorted(prior_keys - inj_keys)

        self.logger.info(f"INJ keys dropped (not in priors): {dropped}")
        self.logger.info(f"PRIOR keys missing from injection dict: {missing}")
    def _bad_prior_support_test(self, likelihood,inj):
        logp = 0.0
        bad = []
        for k, v in inj.items():
            lp = likelihood.priors[k].ln_prob(v)
            if not np.isfinite(lp):
                bad.append((k, v, lp))
            logp += lp

        self.logger.info(f"logPrior(inj) = {logp}")
        if bad:
            self.logger.warning(f"Non-finite ln_prob at injection: {bad[:10]}")
        
    def _probe_local_logl(self,likelihood, theta0, params, rel_scales, n=200, seed=0, logger=None):
        """
        Diagnostic: random local probing around theta0.
        - params: list of parameter names to perturb
        - rel_scales: dict {param: relative step size} e.g. 1e-3, 1e-2
        (absolute step will be derived from prior width if possible, else from |theta0|+1)
        Returns summary dict + samples list.
        """
        rng = np.random.default_rng(seed)

        # Ensure we only use keys relevant to this likelihood
        theta0 = {k: v for k, v in theta0.items() if k in likelihood.priors}

        # Build absolute scales using priors when available (Uniform priors are easiest)
        abs_scales = {}
        for p in params:
            if p not in theta0:
                continue
            prior = likelihood.priors.get(p, None)
            s_rel = rel_scales.get(p, 1e-3)

            s_abs = None
            # Try to infer a characteristic width from the prior
            if hasattr(prior, "minimum") and hasattr(prior, "maximum"):
                width = float(prior.maximum - prior.minimum)
                s_abs = s_rel * width
            if s_abs is None:
                s_abs = s_rel * (abs(float(theta0[p])) + 1.0)

            abs_scales[p] = s_abs

        # Reference logL
        likelihood.parameters.update(theta0)
        logL0 = float(likelihood.log_likelihood())

        records = []
        for i in range(n):
            theta = dict(theta0)
            for p, s_abs in abs_scales.items():
                theta[p] = float(theta[p] + rng.normal(0.0, s_abs))

            likelihood.parameters.update(theta)
            logL = float(likelihood.log_likelihood())
            records.append((logL, logL - logL0, theta))

        # Summaries
        dlogL = np.array([r[1] for r in records], dtype=float)
        summary = {
            "logL0": logL0,
            "dlogL_mean": float(np.mean(dlogL)),
            "dlogL_std": float(np.std(dlogL, ddof=1)) if len(dlogL) > 1 else 0.0,
            "dlogL_min": float(np.min(dlogL)),
            "dlogL_max": float(np.max(dlogL)),
            "frac_improving": float(np.mean(dlogL > 0)),
        }

        if logger is not None:
            logger.info(f"[local probe] logL0={summary['logL0']:.3f}")
            logger.info(f"[local probe] ΔlogL: mean={summary['dlogL_mean']:.3f}, std={summary['dlogL_std']:.3f}, "
                        f"min={summary['dlogL_min']:.3f}, max={summary['dlogL_max']:.3f}, "
                        f"frac(ΔlogL>0)={summary['frac_improving']:.3f}")

            # Log the best point found locally (purely diagnostic)
            best = max(records, key=lambda r: r[0])
            logger.info(f"[local probe] best local logL={best[0]:.3f} (Δ={best[1]:.3f})")
            logger.info(f"[local probe] best local theta (subset): " +
                        ", ".join([f"{p}={best[2].get(p)}" for p in params if p in best[2]]))

        return summary, records
    def _probe_local_logl_test(self,like,theta_inj,theta_ml):
        
        # Example: probe only a few sensitive params first (diagnostic)
        params = ["geocent_time", "phase", "chirp_mass", "mass_ratio", "luminosity_distance"]
        rel_scales = {
            "geocent_time": 1e-3,          # interpreted relative to prior width if Uniform
            "phase": 1e-2,
            "chirp_mass": 1e-3,
            "mass_ratio": 1e-3,
            "luminosity_distance": 1e-3,
        }

        self.probe_local_logl(like, theta_inj, params, rel_scales, n=200, seed=1, logger=self.logger)
        self.probe_local_logl(like, theta_ml,  params, rel_scales, n=200, seed=2, logger=self.logger)
    def _eval_logL_for_ifos(self, ifos_subset, theta):
        like = self.likelihood(ifos_override=ifos_subset)  # you may need to implement this override
        theta = {k: v for k, v in theta.items() if k in like.priors}
        like.parameters.update(theta)
        return float(like.log_likelihood())
    
    def _run_ifo_tests(self,theta_inj,theta_ml):
        ifos_ce = self.scenario.ifos[0:2]
        ifos_et = self.scenario.ifos[2:3]

        logL_inj_ce = self._eval_logL_for_ifos(ifos_ce, theta_inj)
        logL_inj_et = self._eval_logL_for_ifos(ifos_et, theta_inj)
        logL_inj_all = self._eval_logL_for_ifos(ifos_ce + ifos_et, theta_inj)

        # same at ML
        logL_ml_ce  = self._eval_logL_for_ifos(ifos_ce, theta_ml)
        logL_ml_et  = self._eval_logL_for_ifos(ifos_et, theta_ml)
        logL_ml_all = self._eval_logL_for_ifos(ifos_ce + ifos_et, theta_ml)

        self.logger.info(f"logL(inj): CE={logL_inj_ce:.3f}, ET={logL_inj_et:.3f}, ALL={logL_inj_all:.3f}")
        self.logger.info(f"logL(ml):  CE={logL_ml_ce:.3f}, ET={logL_ml_et:.3f}, ALL={logL_ml_all:.3f}")

    def _ML_inj_comparison_test(self,likelihood,inj,result):
        #tests for set-up function
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
        
    def log_diagnostic_tests(self,result):
        #tests for set-up function
        likelihood  = self.likelihood()
        inj = self.scenario.injct_params_waves[0].copy()
        inj = {k: v for k, v in inj.items() if k in likelihood.priors}
        
        theta_inj = self.scenario.injct_params_waves[0].copy()
        theta_ml  = self._getMaximumLikelihood(result)
        
        self._missing_dropped_keys_test(likelihood)
        self._bad_prior_support_test(likelihood,inj)
        self._probe_local_logl_test(likelihood,result,theta_inj,theta_ml)
        self._run_ifo_tests(theta_inj,theta_ml)
        self._ML_inj_comparison_test(likelihood,inj,result)

    def getPrior(self):
        return self.GetSinglePrior()
