from abc import ABC, abstractmethod
import bilby
from bilby.core.result import read_in_result
import copy
from scenario import GWScenario
from .MethodConfig import MethodConfig
from .Method_type import Method_type
import numpy as np
from bilby.gw.detector import InterferometerList
from bilby.core.prior import DeltaFunction
from bilby.gw.conversion import generate_posterior_samples_from_marginalized_likelihood
from bilby.gw.utils import noise_weighted_inner_product
class Method(ABC):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):

        self.run_sampler       = run_sampler
        self.scenario          = scenario
        self.logger            = logger
        self.method_type       = None
        self.config            = config
        self.start_from_chekpt = start_from_chekpt
        self.results           = None
        self.prior             = self.getPrior()
        self.likelihood        = self.getLikelihood()
        
    def overideLikelihood(self,ifos_override):
        #overrirde the likelihood
        self.likelihood = self.getLikelihood(ifos_override)
    
    def getLikelihood(self, ifos_override=None):
        self.logger.info("$$$ get a single likelihood signal")
        #allow to overwride the IFOS for debugging/testing
        if ifos_override is None:
            ifos = self.scenario.ifos
        else:
            # Ensure bilby receives an InterferometerList (works with list slices too)
            if isinstance(ifos_override, InterferometerList):
                ifos = ifos_override
            else:
                ifos = InterferometerList(ifos_override)
        
        priors = self.prior
        if priors is None:
            raise NotImplementedError("prior is not implemented")
        
        fiducial_parameters = self.scenario.injct_params_waves[0].copy()
        fiducial_parameters["time_jitter"] = 0.0
        
        # likelihood = bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient(  #GravitationalWaveTransient
        #     interferometers          = ifos,
        #     waveform_generator       = self.scenario.wg_rel,
        #     priors                   = priors,
        #     fiducial_parameters      = fiducial_parameters,
        #     update_fiducial_parameters=True,
        #     distance_marginalization = False,
        #     phase_marginalization    = True,
        #     time_marginalization     = True,
        #     jitter_time              = False
        # )
        # only set up for debugging
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos,
            waveform_generator=self.scenario.wg,  # NOT wg_rel
            priors=priors,
            distance_marginalization=False,
            phase_marginalization=True,
            time_marginalization=True,
            jitter_time=False,
        )
        return likelihood
    
    def sampeler(self):
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty")
        
        if self.run_sampler:
            clean = not self.start_from_chekpt  #do we need to clean the code
        else:
            clean = False
            
        priors = self.prior
        if priors is None:
            raise NotImplementedError("prior is not implemented")
        
        sample = bilby.run_sampler(
            likelihood = self.likelihood,
            priors     = priors,
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
            outdir     = "logs/log_ET_dynesty_" + self.method_type.code,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    def generateSamples(self):
        self.logger.info("$$$ run the samples")
        #bayesian part
        if self.run_sampler:
            result = self.sampeler()
        else:
            outdir = "logs/log_ET_dynesty_" + self.method_type.code + "/" + self.method_type.code + "_result.json"
            result = read_in_result(outdir) #outdir is also used in sampeler 
        self.updateResults(result)
        
    def updateResults(self,result):
        # method used for postprocessing the result
        raise NotImplementedError

    def UpdateMargPosterior(self,result):
        result.posterior = generate_posterior_samples_from_marginalized_likelihood(
            samples=result.posterior,      # what you read from HDF5
            likelihood=self.likelihood,     # rebuilt likelihood with marg flags enabled
            npool=18,                  # match your compute setting if you like
            block=50,
            use_cache=True,
        )

    ###   priors   ###
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

    def _getMaximumLikelihood(self,result):
        posterior = result.posterior
        idx_ml    = posterior["log_likelihood"].idxmax()
        ml_sample = posterior.loc[idx_ml]
        return ml_sample 
        
    def getResidualIfos_freq(self,MLPosteriorA):
        """returns the residual ifos without the MLPosterior waveform, 
        used for residual analysis of the method.
        Uses the frequency domain"""
        second_wave_ifos     = []
        for ifo in self.scenario.ifos:
            #polarizations
            pols                 = self.scenario.wg.frequency_domain_strain(parameters=dict(MLPosteriorA)) #returns cross and plus waveform
            h_fd                 = ifo.get_detector_response(pols, dict(MLPosteriorA))
            d_fd                 = ifo.strain_data.frequency_domain_strain
            res_fd               = d_fd - h_fd
            second_wave_ifo      = self._clone_ifo_with_new_fd_strain(ifo,res_fd)
            second_wave_ifos.append(second_wave_ifo)
        return InterferometerList(second_wave_ifos)
    
    def _clone_ifo_with_new_fd_strain(self, ifo, new_fd):
        """
        Diagnostic: return an IFO that is identical to `ifo` in every way,
        except that its frequency_domain_strain is replaced by `new_fd`.

        This avoids losing geometry/calibration/min-max-freq/windowing metadata
        that you would lose with get_empty_interferometer().
        """
        if new_fd.shape != ifo.strain_data.frequency_domain_strain.shape:
            raise ValueError(
                f"FD strain shape mismatch for {ifo.name}: "
                f"new_fd {new_fd.shape} vs original {ifo.strain_data.frequency_domain_strain.shape}"
            )
        new_ifo = copy.deepcopy(ifo)
        new_ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=np.array(new_fd, copy=True),
            sampling_frequency=ifo.strain_data.sampling_frequency,
            duration=ifo.strain_data.duration,
            start_time=ifo.strain_data.start_time,
        )
        new_ifo.strain_data.frequency_domain_strain = np.array(new_fd, copy=True)
        return new_ifo
    
    def getResidualIfos_time(self,waveForms,amplitudes,ifo):
        #returns the static likelihood of the residual with amplitudes and waveforms, can be used for model validation and sampling of amplitudes.
        if len(waveForms) != len(amplitudes):
            raise AssertionError(f"both waveForms and amplitudes must have the same lenght but got {len(waveForms)} waveforms and {len(amplitudes)} amplitudes")
        ifo_copy        = copy.deepcopy(ifo)
        residual_strain = ifo_copy.strain_data.time_domain_strain
        for i,waveForm in enumerate(waveForms):
            residual_strain = residual_strain - amplitudes[i]*waveForm
            
        #set the strain to allow the internal workings of bilby to convert to freq domain
        ifo_copy.strain_data.set_from_time_domain_strain(
            time_domain_strain = residual_strain,
            sampling_frequency=ifo.strain_data.sampling_frequency,
            duration=ifo.strain_data.duration,
            start_time=ifo.strain_data.start_time,
        )
        
        #we use the discretized version of the inner product integral (for which we need df and mask)
        psd_array   = ifo_copy.power_spectral_density_array
        residua_feq = ifo_copy.strain_data.frequency_domain_strain
        df          = ifo_copy.frequency_array[1] - ifo_copy.frequency_array[0] #get frequency bin size
        mask        = ifo_copy.frequency_mask #get actual used frequencies

        rr = noise_weighted_inner_product(
            residua_feq[mask],
            residua_feq[mask],
            psd_array[mask],
            df,
        )
        logl = -0.5 * rr
        return ifo_copy,float(logl)
    
    def _zero_fd_waveform(self,frequency_array, **params):
        # Return dict with the polarizations expected by bilby: {'plus': ..., 'cross': ...}
        # For "no signal", both are zeros.
        z = np.zeros_like(frequency_array, dtype=complex)
        return {'plus': z, 'cross': z}
    
    @abstractmethod
    def getPrior(self):
        pass
