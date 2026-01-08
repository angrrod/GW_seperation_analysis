import bilby
from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario
import numpy as np
import time
import os
class SingleSignalMethod(Method):
    def __init__(self, run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):
        super().__init__( run_sampler, scenario, logger, config,start_from_chekpt)
        self.method_type = Method_type.SINGLE
        self.nameExtra = ""
        self._logl_diag_state = {"did_header": set()}
    
    def sampeler(self):
        #adapt the prior
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty for single signal")
        prior = self.prior      
        if self.run_sampler:
            clean = not self.start_from_chekpt  #do we need to clean the code
        else:
            clean = False
        seed_theta = self.scenario.injct_params_waves[0]   # injection dict
        #TODO:delete
        like = self.likelihood
        # live_points = self._make_seeded_live_points_dynesty(
        #     likelihood=like,
        #     priors=prior,
        #     seed_theta=seed_theta,
        #     nlive=self.config.nlive,
        #     seed_frac=0.05,        # 10% of live points near injection
        #     rel_jitter=1e-3        # jitter scale relative to prior width
        # )
        sample = bilby.run_sampler(
            likelihood = like,
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
            # live_points= live_points,
            outdir     = "logs/log_ET_dynesty_" + self.method_type.code + self.nameExtra,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    # diagnostic tests
    def _missing_dropped_keys_test(self,likelihood):
        # dropped or missing keys?
        self.logger.info("$$$ checking for dropped keys")
        inj_full = self.scenario.injct_params_waves[0].copy()
        prior_keys = set(likelihood.priors.keys())
        inj_keys = set(inj_full.keys())

        dropped = sorted(inj_keys - prior_keys)
        missing = sorted(prior_keys - inj_keys)

        self.logger.info(f"$$$ INJ keys dropped (not in priors): {dropped}")
        self.logger.info(f"$$$ PRIOR keys missing from injection dict: {missing}")
        
    def _bad_prior_support_test(self, likelihood,inj):
        self.logger.info("$$$ checking for prior support")
        logp = 0.0
        bad = []
        for k, v in inj.items():
            lp = likelihood.priors[k].ln_prob(v)
            if not np.isfinite(lp):
                bad.append((k, v, lp))
            logp += lp

        self.logger.info(f"$$$ logPrior(inj) = {logp}")
        if bad:
            self.logger.warning(f"$$$ Non-finite ln_prob at injection: {bad[:10]}")
        
    def _probe_local_logl(self,likelihood, theta0, params, rel_scales, n=200, seed=0):
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

        self.logger.info(f"$$$ [local probe] logL0={summary['logL0']:.3f}")
        self.logger.info(f"$$$ [local probe] ΔlogL: mean={summary['dlogL_mean']:.3f}, std={summary['dlogL_std']:.3f}, "
                    f"$$$ min={summary['dlogL_min']:.3f}, max={summary['dlogL_max']:.3f}, "
                    f"$$$ frac(ΔlogL>0)={summary['frac_improving']:.3f}")

        # Log the best point found locally (purely diagnostic)
        best = max(records, key=lambda r: r[0])
        self.logger.info(f"$$$ [local probe] best local logL={best[0]:.3f} (Δ={best[1]:.3f})")
        self.logger.info(f"$$$ [local probe] best local theta (subset): " +
                    ", ".join([f"{p}={best[2].get(p)}" for p in params if p in best[2]]))

        return summary, records
    def _probe_local_logl_test(self,like,theta_inj,theta_ml):
        self.logger.info("$$$ probing likelihood test")
        # Example: probe only a few sensitive params first (diagnostic)
        params = ["geocent_time", "phase", "chirp_mass", "mass_ratio", "luminosity_distance"]
        rel_scales = {
            "geocent_time": 1e-3,          # interpreted relative to prior width if Uniform
            "phase": 1e-2,
            "chirp_mass": 1e-3,
            "mass_ratio": 1e-3,
            "luminosity_distance": 1e-3,
        }
        
        for i in range(10):
            self._probe_local_logl(like, theta_inj, params, rel_scales, n=200, seed=i+1)


    def _eval_logL_for_ifos(self, ifos_subset, theta):
        like = self.getLikelihood(ifos_override=ifos_subset)  # you may need to implement this override
        theta = {k: v for k, v in theta.items() if k in like.priors}
        like.parameters.update(theta)
        return float(like.log_likelihood())
    
    def _run_ifo_tests(self,theta_inj,theta_ml):
        self.logger.info("$$$ check ifos")
        ifos_ce = self.scenario.ifos[0:3]
        ifos_et = self.scenario.ifos[3:5];

        logL_inj_ce = self._eval_logL_for_ifos(ifos_ce, theta_inj)
        logL_inj_et = self._eval_logL_for_ifos(ifos_et, theta_inj)
        logL_inj_all = self._eval_logL_for_ifos(ifos_ce + ifos_et, theta_inj)

        # same at ML
        logL_ml_ce  = self._eval_logL_for_ifos(ifos_ce, theta_ml)
        logL_ml_et  = self._eval_logL_for_ifos(ifos_et, theta_ml)
        logL_ml_all = self._eval_logL_for_ifos(ifos_ce + ifos_et, theta_ml)

        self.logger.info(f"$$$ logL(inj): CE={logL_inj_ce:.3f}, ET={logL_inj_et:.3f}, ALL={logL_inj_all:.3f}")
        self.logger.info(f"$$$ logL(ml):  CE={logL_ml_ce:.3f}, ET={logL_ml_et:.3f}, ALL={logL_ml_all:.3f}")
    
    def getLikelihood(self, ifos_override=None):
        # build the real likelihood object from Method
        like = super().getLikelihood(ifos_override=ifos_override)

        # attach diagnostic logging to this likelihood object
        theta_inj = self.scenario.injct_params_waves[0].copy()
        
        like = self.attach_logl_wrapper(
            like,
            theta_actual=theta_inj,
            every=1000,          # choose cadence
            label_actual="inj",
        )
        return like

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

        self.logger.info(f"$$$ logL(inj)  = {logL_inj:.3f}")
        self.logger.info(f"$$$ logL(ml)   = {logL_ml:.3f}")
        self.logger.info(f"$$$ ΔlogL      = {(logL_ml-logL_inj):.3f}")

        self.logger.info(f"$$$ logLR(inj) = {logLR_inj:.3f}")
        self.logger.info(f"$$$ logLR(ml)  = {logLR_ml:.3f}")
        self.logger.info(f"$$$ ΔlogLR     = {(logLR_ml-logLR_inj):.3f}")
        
        logLR_max = float(result.log_likelihood_evaluations.max())
        logL_noise = float(result.log_noise_evidence)   # this is log L_noise
        logL_full = logLR_max + logL_noise
        
        self.logger.info(f"$$$ logLR_max = {logLR_max}, logL_noise = {logL_noise}, logL_full = {logL_full}")
        
    def log_diagnostic_tests(self,result):
        #tests for set-up function
        likelihood  = self.likelihood
        inj = self.scenario.injct_params_waves[0].copy()
        inj = {k: v for k, v in inj.items() if k in likelihood.priors}
        
        theta_inj = self.scenario.injct_params_waves[0].copy()
        theta_ml  = self._getMaximumLikelihood(result)
        
        self._missing_dropped_keys_test(likelihood)
        self._bad_prior_support_test(likelihood,inj)
        self._probe_local_logl_test(likelihood,theta_inj,theta_ml)
        # self._run_ifo_tests(theta_inj,theta_ml)
        self._ML_inj_comparison_test(likelihood,inj,result)
        
    def _filter_theta_to_priors(self, like, theta: dict) -> dict:
        """Keep only keys that are in the likelihood priors."""
        if theta is None:
            return {}
        priorkeys = set(like.priors.keys())
        return {k: float(v) for k, v in theta.items() if k in priorkeys}

    def _swap_parameters_temporarily(self, like, theta_new: dict):
        """
        Context-like helper: overwrite only the keys in theta_new and restore them afterwards.
        Returns a restore dict that must be applied back.
        """
        restore = {}
        for k, v in theta_new.items():
            # store old value if present, else mark missing
            if k in like.parameters:
                restore[k] = like.parameters[k]
            else:
                restore[k] = None  # sentinel for "missing"
            like.parameters[k] = v
        return restore

    def _restore_parameters(self, like, restore: dict):
        """Undo _swap_parameters_temporarily."""
        for k, old in restore.items():
            if old is None:
                # key didn't exist before; remove if present
                if k in like.parameters:
                    del like.parameters[k]
            else:
                like.parameters[k] = old

    def attach_logl_wrapper(self, like, theta_actual: dict, every: int = 5000, label_actual: str = "actual"):
        """
        Diagnostic wrapper for dynesty sampling:
        - every `every` calls, log current logL, logL(theta_actual), and delta.
        Notes:
        - This will evaluate the likelihood twice every `every` calls.
        - theta_actual should be in the same parameterization as like.priors (e.g. your injection dict).
        """
        orig_logl = like.log_likelihood
        orig_loglr = getattr(like, "log_likelihood_ratio", None)
        counter = {"logl": 0, "loglr": 0}
        best = {"logL": -np.inf, "n": 0, "which": None}
        t0 = time.time()

        theta_actual_f = self._filter_theta_to_priors(like, theta_actual)

        def _wrapped_core(eval_fn, which, *args, **kwargs):
            self._header_once(like,self._logl_diag_state,which)
            logL_cur = float(eval_fn(*args, **kwargs))
            counter[which] += 1
            n = counter[which]
            # Track best (helps sanity check)
            if logL_cur > best["logL"]:
                best["logL"] = logL_cur
                best["n"] = n
                best["which"] = which
                
            # 2) Every N calls, also evaluate reference "actual" logL
            if every is not None and every > 0 and (n % every) == 0:
                # Save/restore only touched keys to avoid side-effects
                restore = self._swap_parameters_temporarily(like, theta_actual_f)
                try:
                    logL_act = float(eval_fn(*args, **kwargs))
                finally:
                    self._restore_parameters(like, restore)

                d = logL_cur - logL_act
                dt = time.time() - t0

                # Log a compact subset of parameters too (optional)
                p = like.parameters
                def _g(k):
                    v = p.get(k, None)
                    return "None" if v is None else f"{float(v):.6g}"

                pid = os.getpid()

                msg = (
                    f"$$$ [logl_wrap] which={which} pid={pid} n={n} t={dt:.1f}s "
                    f"val_cur={logL_cur:.3f} val_{label_actual}={logL_act:.3f} Δ={d:.3f} "
                    f"(best={best['logL']:.3f} @n={best['n']} via {best.get('which','?')}) "
                    f"cm={_g('chirp_mass')} q={_g('mass_ratio')} tc={_g('geocent_time')} "
                    f"phi={_g('phase')} dL={_g('luminosity_distance')}"
                )

                # normal logger (may be swallowed by multiprocessing)
                self.logger.info(msg)

                # file logger (always works under multiprocessing)
                os.makedirs("logs", exist_ok=True)
                with open("logs/logl_test.txt", "a", buffering=1) as f:
                    f.write(msg + "\n")

            return logL_cur

        # def wrapped_log_likelihood(*args, **kwargs):
        #     return _wrapped_core(orig_logl, "logl", *args, **kwargs)

        # like.log_likelihood = wrapped_log_likelihood

        # if orig_loglr is not None:
        #     def wrapped_log_likelihood_ratio(*args, **kwargs):
        #         return _wrapped_core(orig_loglr, "loglr", *args, **kwargs)
        #     like.log_likelihood_ratio = wrapped_log_likelihood_ratio
            
        return like
    
    def _make_seeded_live_points_dynesty(
        self,
        likelihood,
        priors,
        seed_theta: dict,
        nlive: int,
        seed_frac: float = 0.90,
        rel_jitter: float = 1e-6,
    ):
        import numpy as np
        import bilby

        # 1) dynesty samples only "search" parameters (exclude constraints)
        # Priors may contain Constraint objects (mass_1/mass_2) -> must exclude from sampling keys
        sampled_keys = []
        for k in list(getattr(priors, "non_fixed_keys", priors.keys())):
            p = priors[k]
            if isinstance(p, bilby.core.prior.Constraint):
                continue
            sampled_keys.append(k)

        ndim = len(sampled_keys)
        if ndim == 0:
            raise ValueError("No sampled parameters found (ndim=0). Check priors/non_fixed_keys.")

        # 2) Build baseline live_u ~ Uniform(0,1)^ndim
        rng = np.random.default_rng()
        live_u = rng.random((nlive, ndim))

        # 3) Replace a fraction with jittered unit-cube coordinates around the seed point
        n_seed = max(1, int(seed_frac * nlive))
        seed_u0 = np.empty(ndim, dtype=float)

        for j, k in enumerate(sampled_keys):
            p = priors[k]
            if k in seed_theta:
                u = p.cdf(seed_theta[k])
                # Some priors may not implement cdf reliably -> fallback to random
                if u is None or not np.isfinite(u):
                    u = rng.random()
            else:
                u = rng.random()
            # keep strictly inside (0,1) to avoid edge pathology
            u = min(max(float(u), 1e-12), 1.0 - 1e-12)
            seed_u0[j] = u

        # jitter scale in unit cube (small)
        sigma_u = rel_jitter
        seeded_u = seed_u0 + rng.normal(0.0, sigma_u, size=(n_seed, ndim))
        seeded_u = np.clip(seeded_u, 1e-12, 1.0 - 1e-12)

        # overwrite last n_seed rows
        live_u[-n_seed:, :] = seeded_u

        # 4) Map to physical space using Bilby prior transform
        # priors.rescale expects dict-like or array depending on Bilby version; robust method:
        # build v by transforming each coordinate with prior.rescale
        live_v = np.zeros_like(live_u)
        for j, k in enumerate(sampled_keys):
            live_v[:, j] = np.array([priors[k].rescale(u) for u in live_u[:, j]], dtype=float)

        # 5) Evaluate log-likelihood at each live point (must set likelihood.parameters)
        live_logl = np.empty(nlive, dtype=float)
        for i in range(nlive):
            theta_i = {k: float(live_v[i, j]) for j, k in enumerate(sampled_keys)}
            likelihood.parameters.update(theta_i)
            live_logl[i] = float(likelihood.log_likelihood_ratio())

        return (live_u, live_v, live_logl)

    def _header_once(self,like,state,which = 'log_likelihood_ratio'):
        pid = os.getpid()
        if pid in state["did_header"]:
            return
        state["did_header"].add(pid)

        ifos = getattr(like, "interferometers", None)
        ifo_names = [ifo.name for ifo in ifos] if ifos is not None else None

        # frequency array fingerprint (first few and last few values)
        fa = None
        try:
            fa = ifos[0].frequency_array
            fa_fingerprint = (float(fa[0]), float(fa[1]), float(fa[-2]), float(fa[-1]), len(fa))
        except Exception:
            fa_fingerprint = None

        # noise evidence used by logLR (should be constant)
        noise_ev = getattr(like, "noise_log_likelihood", None)
        try:
            noise_ev = float(noise_ev) if noise_ev is not None else None
        except Exception:
            pass

        # data fingerprint
        data_fp = []
        if ifos is not None:
            for ifo in ifos:
                sd = ifo.strain_data
                data_fp.append((ifo.name, float(sd.duration), float(sd.sampling_frequency), float(sd.start_time)))

        # write per-PID file
        path = f"logs/like_fingerprint_pid{pid}.txt"
        with open(path, "a", buffering=1) as f:
            f.write(
                f"[fingerprint] pid={pid} which={which} "
                f"ifo_names={ifo_names} data={data_fp} "
                f"freq_fp={fa_fingerprint} noise_logL={noise_ev}\n"
            )
    def getPrior(self):
        return self.GetSinglePrior() 
