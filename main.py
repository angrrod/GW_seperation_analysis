import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector import get_empty_interferometer, InterferometerList
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from bilby.core.result import read_in_result
from jointRB.JointLikelihoodRB import OverlappingSignalsRelBinning
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, asdict
import copy
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters

### this file consists of the main loop for the tests conscerning GW separation analysis ###
@dataclass(frozen=True)
class WaveformConfig:
    waveform_approximant: str  = "IMRPhenomPv2"
    minimum_frequency: float   = 10.0
    sampling_frequency: float  = 4096.0
    reference_frequency: float = 10.0
    duration: float            = 10.0
    ASD_file_name: str         = "ET_C" #TODO: get ET_D config

class GWScenario:
    def __init__(self, logger ,config : WaveformConfig, injct_params_waves = None):
        self.logger             = logger
        self.config             = config
        self._prior              = self.getJointPriors()
        if injct_params_waves is None:
            self.injct_params_waves = self.GetWaveFormParams()
            
        #parameters to be initialized during set_up:
        self.ifo      = None
        self.noise_td = None #for plotting
        self.wg       = None
        
    
    @property
    #make the prior immutable
    def prior(self): return self._prior
    
    def setUpScenario(self):
        """
        generates the waveform and interferrometer data structures
        
        """
        self.logger.info("$$$ setting up scenario; generating overlapping waves")
        
        #load realistic background and PSD
        asd_f, asd = np.loadtxt(self.config.ASD_file_name+".txt", unpack=True)  
        psd        = asd**2
        np.savetxt(self.config.ASD_file_name+"_PSD"+".txt", np.column_stack([asd_f, psd]))
            
        self.ifo = bilby.gw.detector.get_empty_interferometer("L1") # TODO: make einstein like
        
        #load spectral density according to the file
        self.ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
            psd_file=self.config.ASD_file_name+"_PSD"+".txt"
        )

        #add gaussian noise
        self.ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=self.config.sampling_frequency,
            duration=self.config.duration,
            start_time=0.0
        )
        
        #noise background used for plotting
        self.noise_td = self.ifo.strain_data.time_domain_strain.copy()
        
        # BBH signal
        self.wg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.config.duration,
            sampling_frequency=self.config.sampling_frequency,
            frequency_domain_source_model=lal_binary_black_hole,
            waveform_arguments={k: v for k, v in asdict(self.config).items() if k not in {"sampling_frequency","duration","ASD_file_name"}}  #remove redundant variables
        )
        
        #inject N waves
        self.logger.info("$$$ injecting " + str(len(self.injct_params_waves)) + " waves")
        for inject_params in self.injct_params_waves:
            self.ifo.inject_signal(
                waveform_generator=self.wg,
                parameters=inject_params
            )
    
    def _getDataTimeSeries(self):
        """
        get timeseries objects of original noise and signal+noise for plotting 
        
        Args:
            sampling_frequency (_type_): _description_
        Returns:
            ts: original timeseries with noise + signal
            ts_noise: timeseries noise 
        """
        self.logger.info("$$$ getting time series objects")
        td = self.ifo.strain_data.time_domain_strain          # numpy array (length = duration * fs)
        t0 = self.ifo.strain_data.start_time                  # GPS start time (float)
        fs = self.config.sampling_frequency

        #convert data
        ts       = TimeSeries(td, sample_rate=fs, epoch=t0)
        ts_noise = TimeSeries(self.noise_td, dt=1/fs, epoch=t0)
        return ts, ts_noise
    
    ### plots ###
    def _PlotTimeSignal(self,ts,timeCenter,ts_noise,fileName):
        #plot time domain of signal
        self.logger.info("$$$ making time domain plot")
        white   = ts.whiten(4, 2).bandpass(40, 200)
        white_noise = ts_noise.whiten(4, 2).bandpass(40, 200)
        t_start = timeCenter - 1.0
        t_end   = timeCenter + 0.2

        white_zoom    = white.crop(t_start, t_end)
        white_noise_zoom = white_noise.crop(t_start, t_end)
        
        #rescale
        pure_signal = white_zoom.value - white_noise_zoom.value
        white = white_zoom.value

        raw_max = np.max(np.abs(pure_signal))
        white_max = np.max(np.abs(white))

        scale = white_max / raw_max        # bring raw up to whitened level
        raw_scaled = pure_signal * scale
        
        # Plot manually using matplotlib (allows full control)
        fig, ax = plt.subplots(figsize=(8, 4))

        # Whitened signal (red, more prominent)
        ax.plot(white_zoom.times, white_zoom.value,
                color="red", alpha=0.1, lw=1.2, label="full signal")
        # Raw data (grey, semi-transparent)
        ax.plot(white_zoom.times, raw_scaled,
                color="black", alpha=1, lw=1, label="signal")
        # Styling
        ax.set_title("ET1 strain: time domain")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Strain")
        ax.legend(loc="upper right")

        # Save
        fig.savefig("Plots/"+fileName+".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        
    def _PlotQtrans(self,timeCenter,ts,fileName):
        self.logger.info("$$$ making Qtransformed plot")
        qspec = ts.q_transform(
            qrange=(8, 8),
            frange=(20, 512),
            outseg=(timeCenter - 4, timeCenter + 4), 
            whiten=True,              
        )

        # Let GWpy handle the plotting
        fig = qspec.plot()
        ax = fig.gca()
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Q-transform of strain around first merger")
        fig.savefig("Plots/"+fileName+".png", dpi=300, bbox_inches="tight")
        
    def makePlots(self,fileNames):
        self.logger.info("$$$ making plots")
        # data plots
        # corner plot already done
        self.logger.info("$$$ making plots")
        ts, ts_noise = self._getDataTimeSeries()
        tc           = self.injct_params_waves[0]['geocent_time']
        
        self._PlotTimeSignal(ts,tc,ts_noise,fileNames[0])
        self._PlotQtrans(tc,ts,fileNames[1])

    def GetSinglePrior(self):
        self.logger.info("$$$ getting a waveform prior")
        prior = bilby.gw.prior.BBHPriorDict()  #allow for default ranges in ET
        if "geocent_time" not in prior:
            prior["geocent_time"] = bilby.core.prior.Uniform(
                minimum=0.5,#maybe make this a bit bigger?
                maximum=10,  
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
    
    def GetWaveFormParams(self):
        self.logger.info("$$$ getting waveform parameters")
        
        #convert masses to chirp and ratio
        m1_1, m2_1 = 22.0, 20.0
        m1_2, m2_2 = 25.0, 20.0
        chirp_1, q_1 = masses_to_chirp_and_q(m1_1, m2_1)
        chirp_2, q_2 = masses_to_chirp_and_q(m1_2, m2_2)
        
        injct_params_wave_1 = dict(
            chirp_mass=chirp_1,
            mass_ratio=q_1,
            a_1=0.4,  #part of the spin of the black hole
            a_2=0.3,
            tilt_1=0.5, #part of the spin of the black hole
            tilt_2=1.0,
            phi_12=1.7,  #part of the spin of the black hole
            phi_jl=0.3,
            luminosity_distance=1000.0, #2000
            theta_jn=1.4, #angle of angular momentum
            psi=2.659,  #angle of polarization
            phase=1.3,
            geocent_time=5,
            ra=1.375, #longituded
            dec=-1.2108,  #lattiude
        )
        injct_params_wave_2 = dict(
            chirp_mass=chirp_2,
            mass_ratio=q_2,
            a_1=0.4,  #part of the spin of the black hole
            a_2=0.9,
            tilt_1=0.2, #part of the spin of the black hole
            tilt_2=1.0,
            phi_12=5.7,  #part of the spin of the black hole
            phi_jl=0.3,
            luminosity_distance=1000.0, #2000
            theta_jn=1.5, #angle of angular momentum
            psi=2.659,  #angle of polarization
            phase=1.2,
            geocent_time=4.5,
            ra=1.75, #longituded
            dec=-1.8,  #lattiude
        )
        injct_params_waves = [injct_params_wave_1,injct_params_wave_2]
        return injct_params_waves
    
    def build_ref_injection(self,injections):
        """
        Convert [dictA, dictB, ...] → one dict with suffixes _A, _B, ...
        used in the joint likelihood method
        """        
        self.logger.info("$$$ renaming injection parameters")
        joint = {}
        for i, single in enumerate(injections):
            suffix = chr(ord("A") + i)  
            for key, value in single.items():
                joint[f"{key}_{suffix}"] = value

        return joint
class Method(ABC):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger):

        self.run_sampler = run_sampler
        self.scenario    = scenario
        self.logger      = logger
        self.method_type = None
        
        #results
        self.sample = None
    
    def likelihood(self):
        self.logger.info("$$$ get a single likelihood signal")
        prior = self.scenario.GetSinglePrior()
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers= [self.scenario.ifo],
            waveform_generator=self.scenario.wg,
            priors=prior,
            distance_marginalization=False,
            phase_marginalization=False,
            time_marginalization=False,
            # reference_frame="H1L1", #depends on the detector config -> ok?
            # time_reference="H1",
        )
        return likelihood
    
    def sampeler(self,resume):
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty")
        
        prior = self.scenario.prior
        sample = bilby.run_sampler(
            likelihood=self.likelihood(),
            priors=prior,
            sampler="dynesty",
            nlive=50, 
            dlogz=2.0, #stopping criterion for the evidence
            sample="rwalk",  
            walks=10, #steps for MCMC sampeler to select new candidates     
            nact=3, #amount of steps is tuned so autocorr is small enough 
            resume=resume,
            outdir="outdir_ET_dynesty_" + self.method_type.code,
            label=self.method_type.code,
        )
        return sample
    
    def generateSamples(self):
        self.logger.info("$$$ run the samples")
        #bayesian part
        if self.run_sampler:
            full_rerun = not self.run_sampler
            result = self.sampeler(full_rerun)
        else:
            outdir = "outdir_ET_dynesty_" + self.method_type.code + "/" + self.method_type.code + "_result.json"
            result = read_in_result(outdir) #outdir is also used in sampeler 
        self.posterior_sample = result.posterior #pandas data frame of samples
    
class SingleSignalMethod(Method):
    def __init__(self, run_sampler:bool,scenario:GWScenario,logger):
        super().__init__( run_sampler, scenario, logger)
        self.method_type = Method_type.SINGLE
    
    def sampeler(self,resume):
        #adapt the prior
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty for single signal")
        prior = self.scenario.GetSinglePrior()
        sample = bilby.run_sampler(
            likelihood=self.likelihood(),
            priors=prior,
            sampler="dynesty",
            nlive=50, 
            dlogz=2.0, #stopping criterion for the evidence
            sample="rwalk",  
            walks=10, #steps for MCMC sampeler to select new candidates     
            nact=3, #amount of steps is tuned so autocorr is small enough 
            resume=resume,
            outdir="outdir_ET_dynesty_" + self.method_type.code,
            label=self.method_type.code,
        )
        return sample

class JointLikelihoodlMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger):
        super().__init__(run_sampler, scenario, logger)
        self.method_type = Method_type.JOINT
        
    def likelihood(self):
        #adapt both prior and likelihood for joint modelling
        self.logger.info("$$$ get the joint likelihood signal")
        waveform_parms = self.scenario.GetWaveFormParams()
        ref_injection  = self.scenario.build_ref_injection(waveform_parms)
        
        wg_rb          = self._getRBWaveForm()
        likelihood     = OverlappingSignalsRelBinning(
            interferometers    = [self.scenario.ifo],
            waveform_generator = wg_rb,
            ref_injection      = ref_injection, # actual parameters in simulation, ML for actual data, this is the FUDICIAL waveform used in the RB scheme
            N_overlaps         = 2,
            priors             = self.scenario.prior,
            reference_frame    = "sky",
            time_reference     = "geocenter",
            delta              = 0.03,  # RB binning tolerance
        )
        return likelihood
    
    def _getRBWaveForm(self):
        """
            wavform generator tailored for jointRB 
        """
        self.logger.info("$$$ get relative binning waveform generator")
        wg_rb = bilby.gw.waveform_generator.WaveformGenerator(
            duration                      = self.scenario.wg.duration,
            sampling_frequency            = self.scenario.wg.sampling_frequency,
            frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,  #jrb_lal_binary_black_hole self.scenario.wg.frequency_domain_source_model
            waveform_arguments            = self.scenario.wg.waveform_arguments.copy()
        )
        return wg_rb
    
class HyrarchicalMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger):
        super().__init__(run_sampler, scenario, logger, )
        self.method_type     = Method_type.HIERARCHICAL
        self.singleSampler   = SingleSignalMethod(run_sampler, scenario, logger)
        self.second_wave_ifo = self.scenario.ifo
        
    def generateSamples(self):
        self.logger.info("$$$ generate Samples for hyrarchical model")
        posteriorSampleA     = self.singleSampler.generateSamples()
        MLPosteriorA         = getMaximumLikelihood(posteriorSampleA)
        pols                 = self.scenario.wg.frequency_domain_strain(MLPosteriorA) #returns cross and plus waveform
        h_fd                 = self.scenario.ifo.get_detector_response(pols, MLPosteriorA)
        d_fd                 = self.scenario.ifo.strain_data.frequency_domain_strain
        res_fd               = d_fd - h_fd
        second_wave_ifo      = self._GetIfoResidual(res_fd)
        self.second_wave_ifo = second_wave_ifo
        
    def likelihood(self):
        self.logger.info("$$$ get likelihood signal for custom ifo")
        prior = self.scenario.GetSinglePrior()
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers          = [self.second_wave_ifo],
            waveform_generator       = self.scenario.wg,
            priors                   = prior,
            distance_marginalization = False,
            phase_marginalization    = False,
            time_marginalization     = False,
            # reference_frame="H1L1", #depends on the detector config -> ok?
            # time_reference="H1",
        )
        return likelihood

    def _GetIfoResidual(self,res_fd):
        self.logger.info("$$$ get residual interferrometer")
        #build copy for second interferrometer
        new_ifo = get_empty_interferometer(self.scenario.ifo.name)
        new_ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain = res_fd,
            sampling_frequency      = self.scenario.ifo.strain_data.sampling_frequency,
            duration                = self.scenario.ifo.strain_data.duration,
            start_time              = self.scenario.ifo.strain_data.start_time,
        )
        new_ifo.power_spectral_density = self.scenario.ifo.power_spectral_density
        return new_ifo

class Method_type(Enum):
    # SINGLE       = ("single_likl", SingleSignalMethod)
    # JOINT        = ("joint_likl", JointLikelihoodlMethod)
    HIERARCHICAL = ("hierarchical",HyrarchicalMethod)  
    
    def __init__(self, code, method: Method):
        self.code   = code
        self.method = method

### helper functions ###

#TODO:refactor?
#TODO: different center parameter
def getMaximumLikelihood(posterior):
    idx_ml = posterior["log_likelihood"].idxmax()
    ml_sample = posterior.loc[idx_ml]
    return {k: ml_sample[k] for k in posterior.search_parameter_keys} #format for waveform generator
    
def masses_to_chirp_and_q(m1, m2):
    # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
    if m1 < m2:
        m1, m2 = m2, m1
    q = m2 / m1                       # mass_ratio in (0, 1]
    # chirp mass in solar masses
    chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    return chirp, q

###########################
####     Main Loop     ####
###########################

def Main(run_sampler):
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    #Logger
    bilby.core.utils.setup_logger(
        log_level="INFO",
        label="my_run", 
        outdir="out",
    )
    logger = bilby.core.utils.logger
    logger.info("$$$ start_run")
    
    #build scenario
    config = WaveformConfig()
    scenario = GWScenario(logger, config)
    scenario.setUpScenario()
    scenario.makePlots(["strain_time_domain_set_up","qtransform_set_up"])
    
    #TODO make loop for different methods
    for method_type in Method_type:
        logger.info(f"$$$ Running method: {method_type.code}")
        method  = method_type.method(run_sampler,scenario,logger)
        sample  = method.generateSamples()
        # analyze the samples
    
    #post processing
    # injct_params_wave = sample.to_dict(orient="records")[0:1]
    # SetupSignalAndDetector(wfv_args,ASD_file_name,injct_params_wave,True,logger)
    
    #make corner plot of the posterior samples
    # result.plot_corner()

###########################
###   Run actual Code   ###
###########################

Main(run_sampler = True)