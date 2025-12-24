from abc import ABC, abstractmethod
import bilby
from bilby.core.result import read_in_result
import copy
from scenario import GWScenario
from .MethodConfig import MethodConfig
import numpy as np
from bilby.gw.detector import InterferometerList

class Method(ABC):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):

        self.run_sampler       = run_sampler
        self.scenario          = scenario
        self.logger            = logger
        self.method_type       = None
        self.config            = config
        self.start_from_chekpt = start_from_chekpt
        #results
        self.results = None
    
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
    
    def sampeler(self):
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty")
        
        if self.run_sampler:
            clean = not self.start_from_chekpt  #do we need to clean the code
        else:
            clean = False
            
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
        self.results = {"waveFormA" : result} #result object
        
    ###   priors   ###
    def GetSinglePrior(self):
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
            minimum=4, maximum=200, name="chirp_mass"
        )
        prior["mass_ratio"] = bilby.core.prior.Uniform(
            minimum=0.1, maximum=1, name="mass_ratio"
        )
        prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=1e3,      
            maximum=1e5,      
            cosmology='Planck15',
            name='luminosity_distance',
            latex_label='$d_L$',
            unit='Mpc'
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
        return {k: ml_sample[k] for k in result.search_parameter_keys} #format for waveform generator, 
        #"result.search_parameter_keys" can go wrong if you constrain/marginalize some parameters
        
    def getResidualIfos(self,MLPosteriorA):
        """returns the residual ifos without the MLPosterior waveform, used for residual analysis of the method"""
        second_wave_ifos     = []
        for ifo in self.scenario.ifos:
            pols                 = self.scenario.wg.frequency_domain_source_model(ifo.frequency_array,MLPosteriorA) #returns cross and plus waveform
            h_fd                 = ifo.get_detector_response(pols, MLPosteriorA)
            d_fd                 = ifo.strain_data.frequency_domain_strain
            res_fd               = d_fd - h_fd
            second_wave_ifo      = self._clone_ifo_with_new_fd_strain(res_fd,ifo)
            second_wave_ifos.append(second_wave_ifo)
        return InterferometerList(second_wave_ifo)
    
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
        new_ifo.strain_data.frequency_domain_strain = np.array(new_fd, copy=True)

        return new_ifo
