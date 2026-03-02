from abc import ABC, abstractmethod
import bilby
from bilby.core.result import read_in_result
import copy
from scenario import GWScenario
from bilby.core.prior import DeltaFunction
from bilby.core.result import read_in_result
from enum import Enum
from bilby.gw.detector import InterferometerList
from utils import get_base_log_dir
from utils import getMaximumLikelihood
import numpy as np
import time
import os
class RunMode(Enum):
    RUN    = "run"
    REUSE  = "reuse"
    CHEKPT = "checkpoint"
    
class Method(ABC):
    def __init__(self,scenario:GWScenario,logger,config,pipeline_type_code:str):
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
        if self.config.sampler == "dynesty": #delete
            sample = self._dynestySample(resume,clean,likelihood)
        elif self.config.sampler == "numpyro":
            sample = self._NUTSSample(resume,clean,likelihood)
        else:
            raise NotImplementedError("sampling method not implemented")
        results = self.updateResults(sample,likelihood)
        return results
    
    def _dynestySample(self,resume,clean,likelihood):
        sample = bilby.run_sampler(
            likelihood   = likelihood,
            priors       = self.prior,
            sampler      = self.config.sampler,
            nlive        = self.config.nlive, 
            dlogz        = self.config.dlogz, #stopping criterion for the evidence
            sample       = self.config.sample,  
            walks        = self.config.walks, #steps for MCMC sampeler to select new candidates  
            bound        = self.config.bound,
            maxmcmc      = self.config.maxmcmc,
            nact         = self.config.nact, #amount of steps is tuned so autocorr is small enough 
            resume       = resume,
            clean        = clean,
            # live_points= live_points, #TODO: DELETE
            outdir       = str(self.getSampleOutdir(self.nameExtra)),
            label        = self.pipeline_type_code,
            npool        = self.config.cores,
            queue_size   = self.config.cores,
            print_method = 'interval-60'
        )
        return sample
    
    def _NUTSSample(self,resume,clean,likelihood): #delete
        sample = bilby.run_sampler(
            likelihood         = likelihood,
            priors             = self.prior,
            sampler            = self.config.sampler,
            sampler_name       = self.config.sampler_name,
            num_warmup         = self.config.tune,     
            num_samples        = self.config.draws,   
            outdir             = str(self.getSampleOutdir(self.nameExtra)),
            label              = self.pipeline_type_code,
            resume             = resume,
            clean              = clean,
            # target_accept_prob = self.config.target_accept,
            # max_tree_depth     = self.config.max_treedepth,
            npool              = self.config.cores,
            sampler_kwargs=dict(
                target_acceptance = self.config.target_accept,
                max_tree_depth    = self.config.max_treedepth,
            ),
        )
        return sample
    
    def run(self,runMode:RunMode,ifos_override):
        #main entry point
        self.logger.info("$$$ Generating posterior samples")
        if runMode == RunMode.REUSE:
            outdir = str(self.getResultPath())
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
    
    def getSampleOutdir(self,nameExtra):
        base    = get_base_log_dir()
        return base / f"log_dynesty_{self.pipeline_type_code}_{nameExtra}"
    
    def getResultPath(self):
        outdir = self.getSampleOutdir(self.nameExtra)
        return outdir / f"{self.pipeline_type_code}_result.json"


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
        fixed_priors   = ["chirp_mass","geocent_time","mass_ratio","luminosity_distance"]
        prior = self.fixPriors(prior,fixed_priors,waveformIdx,self.config.use_deltas)
        
        #fix priors for waveform
        if self.scenario.config.waveform_approximant == "IMRPhenomD":
            fixed_priors   = ["a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
            prior = self.fixPriors(prior,fixed_priors,waveformIdx,True)
        return prior
    
    ### WJ: 04/01/25 fix priors for debugging
    def fixPriors(self,prior,fixed_priors,waveformIdx,useDeltaFunction):
        waveFormParams = self.scenario.GetWaveFormParams()
        for k in fixed_priors:
            if k in prior:
                value = waveFormParams[waveformIdx].get(k)
                if useDeltaFunction:
                    self.logger.info(f"$$$ making prior delta {k}")
                    prior[k] = DeltaFunction(value, name=k)
                elif self.config.restrict_prior:
                    if (value > prior[k].maximum) or (value < prior[k].minimum):
                        raise ValueError(f'value: {value} not supported in prior [{prior[k].minimum},{prior[k].maximum}]')
                    upperDiff  = prior[k].maximum - value
                    lowerDiff  = value - prior[k].minimum
                    priorRange = (prior[k].maximum - prior[k].minimum)
                    tolerance  = 1e-3
                    delta      = self.config.restriction_str * min(upperDiff,lowerDiff)
                    #deal with boundaries
                    if upperDiff < tolerance:
                        priorMin = value - priorRange*self.config.restriction_str
                        priorMax = value + delta
                    elif lowerDiff < tolerance:
                        priorMin = value - delta
                        priorMax = value + priorRange*self.config.restriction_str
                    else:
                        priorMax = value + delta
                        priorMin = value - delta
                    prior[k].maximum = priorMax
                    prior[k].minimum = priorMin
                    self.logger.info(f"$$$ tightening prior {k}, with bounds: {value + delta} and {value - delta}")
        return prior
    
    # needed for joint parameter estimation
    # independent priors for both
    def getJointPriors(self):
        self.logger.info("$$$ getting joint priors")
        #get waveforms with prior around actual values
        priors   = bilby.core.prior.PriorDict()
        for waveformidx in range(2):
            prior = self.GetSinglePrior(waveformidx)
            for key, prior in prior.items():
                if waveformidx==0: #44-57 for time
                    priors[f"{key}_A"] = copy.deepcopy(prior) 
                elif waveformidx == 1:
                    priors[f"{key}_B"] = copy.deepcopy(prior)
        
        # break prior symmetry, ensure that one signal is later than the other
        max_time_A = self.scenario.GetWaveFormParams()[0].get("geocent_time") #used in symmetry breaking
        max_time_B = self.scenario.GetWaveFormParams()[1].get("geocent_time")
        if max_time_A > max_time_B:
            priors["geocent_time_B"] = bilby.core.prior.Uniform(
                                        minimum=40,
                                        maximum=max_time_A,  
                                        name="geocent_time_B",
                                    )
        else: 
            priors["geocent_time_A"] = bilby.core.prior.Uniform(
                                        minimum=40,
                                        maximum=max_time_B,  
                                        name="geocent_time_A",
                                    )
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







####################################
###   diagnostic tests/Logging   ###
####################################

    @abstractmethod
    def log_diagnostic_tests(self,result,ifos_override):
        pass
    
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
        
    def _probe_local_logl(
        self,
        likelihood,
        theta0: dict,
        params: list[str],
        rel_scales: dict,
        n: int = 200,
        seed: int = 0,
        clamp_to_prior: bool = True,
        assert_consistency: bool = True,
        consistency_tol: float = 1e-3,
    ):
        """
        Diagnostic: random local probing around theta0.

        Changes vs your version:
        1) Sets likelihood.parameters = dict(theta) (not .update) to reduce cache artefacts.
        2) Optionally clamps proposals to bounded priors.
        3) Computes logL, logLR, logL_noise and checks logL ≈ logL_noise + logLR.
        4) Logs marginalisation flags and warns if you probe marginalised parameters.

        - params: list of parameter names to perturb
        - rel_scales: dict {param: relative step size} e.g. 1e-3, 1e-2
        absolute step is derived from prior width if possible, else from |theta0|+1
        Returns: (summary_dict, records)
        where records are dicts with keys: logL, dlogL, logLR, logL_noise, err, theta
        """
        rng = np.random.default_rng(seed)

        # Keep only parameters that are in the priors of this likelihood
        theta0 = {k: v for k, v in theta0.items() if k in likelihood.priors}

        # Log marginalisation status (bilby GW likelihood has these attrs)
        time_marg = getattr(likelihood, "time_marginalization", None)
        phase_marg = getattr(likelihood, "phase_marginalization", None)
        dist_marg = getattr(likelihood, "distance_marginalization", None)
        self.logger.info(
            f"$$$ [local probe] marg flags: time={time_marg}, phase={phase_marg}, dist={dist_marg}"
        )

        # Warn if probing parameters that are typically marginalised
        marginalised_names = {"geocent_time": time_marg, "phase": phase_marg, "luminosity_distance": dist_marg}
        bad = [p for p in params if marginalised_names.get(p, False)]
        if bad:
            self.logger.warning(
                "$$$ [local probe] You are probing parameters that appear to be marginalised in this likelihood: "
                + ", ".join(bad)
                + ". Results may be hard to interpret."
            )

        # Build absolute scales
        abs_scales = {}
        for p in params:
            if p not in theta0:
                continue

            prior = likelihood.priors.get(p, None)
            s_rel = float(rel_scales.get(p, 1e-3))

            s_abs = None
            if prior is not None and hasattr(prior, "minimum") and hasattr(prior, "maximum"):
                try:
                    width = float(prior.maximum - prior.minimum)
                    s_abs = s_rel * width
                except Exception:
                    s_abs = None

            if s_abs is None:
                s_abs = s_rel * (abs(float(theta0[p])) + 1.0)

            abs_scales[p] = float(s_abs)

        # Reference point
        likelihood.parameters = dict(theta0)
        logL0 = float(likelihood.log_likelihood())

        # For consistency checks
        def _maybe_noise_logl():
            fn = getattr(likelihood, "noise_log_likelihood", None)
            return float(fn()) if callable(fn) else None

        def _maybe_loglr():
            fn = getattr(likelihood, "log_likelihood_ratio", None)
            if self.pipeline_type_code == "joint_likl":
                return float(fn(parameters = None)) if callable(fn) else None
            else:
                return float(fn()) if callable(fn) else None

        logL0_noise = _maybe_noise_logl()
        logL0_lr = _maybe_loglr()
        if assert_consistency and (logL0_noise is not None) and (logL0_lr is not None):
            err0 = logL0 - (logL0_noise + logL0_lr)
            if abs(err0) > consistency_tol:
                self.logger.warning(
                    f"$$$ [local probe] Inconsistency at theta0: "
                    f"logL - (noise+logLR) = {err0:.6g}"
                )

        records = []
        for _ in range(n):
            theta = dict(theta0)

            # Propose
            for p, s_abs in abs_scales.items():
                theta[p] = float(theta[p] + rng.normal(0.0, s_abs))

                # Optional clamp to bounded priors
                if clamp_to_prior:
                    prior = likelihood.priors.get(p, None)
                    if prior is not None and hasattr(prior, "minimum") and hasattr(prior, "maximum"):
                        try:
                            lo = float(prior.minimum)
                            hi = float(prior.maximum)
                            if theta[p] < lo:
                                theta[p] = lo
                            elif theta[p] > hi:
                                theta[p] = hi
                        except Exception:
                            pass

            # Evaluate
            likelihood.parameters = dict(theta)
            logL = float(likelihood.log_likelihood())
            logL_noise = _maybe_noise_logl()
            logLR = _maybe_loglr()

            err = None
            if (logL_noise is not None) and (logLR is not None):
                err = float(logL - (logL_noise + logLR))
                if assert_consistency and abs(err) > consistency_tol:
                    # Don't spam: store it; also occasionally warn
                    self.logger.debug(
                        f"$$$ [local probe] Inconsistency: logL - (noise+logLR) = {err:.6g}"
                    )

            records.append(
                {
                    "logL": logL,
                    "dlogL": float(logL - logL0),
                    "logLR": logLR,
                    "logL_noise": logL_noise,
                    "consistency_err": err,
                    "theta": theta,
                }
            )

        # Summaries
        dlogL = np.array([r["dlogL"] for r in records], dtype=float)
        summary = {
            "logL0": logL0,
            "dlogL_mean": float(np.mean(dlogL)),
            "dlogL_std": float(np.std(dlogL, ddof=1)) if len(dlogL) > 1 else 0.0,
            "dlogL_min": float(np.min(dlogL)),
            "dlogL_max": float(np.max(dlogL)),
            "frac_improving": float(np.mean(dlogL > 0)),
        }

        # Add quick consistency diagnostics
        errs = [r["consistency_err"] for r in records if r["consistency_err"] is not None]
        if errs:
            errs = np.array(errs, dtype=float)
            summary["consistency_err_maxabs"] = float(np.max(np.abs(errs)))
            summary["consistency_err_mean"] = float(np.mean(errs))
        else:
            summary["consistency_err_maxabs"] = None
            summary["consistency_err_mean"] = None

        self.logger.info(f"$$$ [local probe] logL0={summary['logL0']:.3f}")
        self.logger.info(
            f"$$$ [local probe] ΔlogL: mean={summary['dlogL_mean']:.3f}, std={summary['dlogL_std']:.3f}, "
            f"min={summary['dlogL_min']:.3f}, max={summary['dlogL_max']:.3f}, "
            f"frac(ΔlogL>0)={summary['frac_improving']:.3f}"
        )
        if summary["consistency_err_maxabs"] is not None:
            self.logger.info(
                f"$$$ [local probe] max|logL-(noise+logLR)| = {summary['consistency_err_maxabs']:.3g}, "
                f"mean err = {summary['consistency_err_mean']:.3g}"
            )

        # Best point
        best = max(records, key=lambda r: r["logL"])
        self.logger.info(
            f"$$$ [local probe] best local logL={best['logL']:.3f} (Δ={best['dlogL']:.3f})"
        )
        if best["logL_noise"] is not None and best["logLR"] is not None:
            self.logger.info(
                f"$$$ [local probe] best: logL_noise={best['logL_noise']:.3f}, "
                f"logLR={best['logLR']:.3f}, "
                f"err={best['consistency_err']:.3g}"
            )

        self.logger.info(
            "$$$ [local probe] best local theta (subset): "
            + ", ".join([f"{p}={best['theta'].get(p)}" for p in params if p in best["theta"]])
        )

        return summary, records

    def _probe_local_logl_test(self, like, theta_inj):
        self.logger.info("$$$ probing likelihood test")

        # If your likelihood marginalises time/phase/distance, do NOT probe them here
        time_marg = getattr(like, "time_marginalization", False)
        phase_marg = getattr(like, "phase_marginalization", False)
        dist_marg = getattr(like, "distance_marginalization", False)

        params = ["chirp_mass", "mass_ratio"]  # safe baseline
        rel_scales = {
            "chirp_mass": 1e-3,
            "mass_ratio": 1e-3,
        }

        # Only probe these if they are NOT marginalised
        if not time_marg:
            params.append("geocent_time")
            rel_scales["geocent_time"] = 1e-3
        if not phase_marg:
            params.append("phase")
            rel_scales["phase"] = 1e-2
        if not dist_marg:
            params.append("luminosity_distance")
            rel_scales["luminosity_distance"] = 1e-3

        for i in range(10):
            self._probe_local_logl(
                like,
                theta_inj,
                params,
                rel_scales,
                n=200,
                seed=i + 1,
                clamp_to_prior=True,
                assert_consistency=True,
                consistency_tol=1e-3,
            )

    def _ML_inj_comparison_test(self,likelihood,inj,result):
        #tests for set-up function
        likelihood.parameters.update(inj)
        logL_inj   = likelihood.log_likelihood()
        logLR_inj  = likelihood.log_likelihood_ratio()

        # ML point from the result
        ml = getMaximumLikelihood(result)   # your helper
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
