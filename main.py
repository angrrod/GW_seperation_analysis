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
import time
from collections import defaultdict
import os
import json
import corner
from matplotlib.lines import Line2D

### this file consists of the main loop for the tests conscerning GW separation analysis ###
@dataclass(frozen=True)
class WaveformConfig:
    waveform_approximant: str  = "IMRPhenomPv2"
    minimum_frequency: float   = 20.0 #10
    sampling_frequency: float  = 4096.0
    reference_frequency: float = 20.0 #10
    duration: float            = 6.0
    ASD_file_name: str         = "ET_D"

@dataclass(frozen=True)
class MethodConfig:
    sampler:str  = "dynesty"
    nlive: int   = 2000 #800
    dlogz: float = 0.1         #stopping criterion for the evidence
    sample: str  = "rslice" #rslice #unif', 'rwalk', 'slice', 'rslice', and 'auto' # performed until the autocorrelation length of the chain can be accurately determined.
    bound: str   = "multi"
    walks: int   = None          #steps for MCMC sampeler to select new candidates     
    nact: int    = 200     #50      #amount of steps is tuned so autocorr is small enough, needed for determining the correct slicing behaviour
    npool: int   = 18
    maxmcmc: int = 20000
class GWScenario:
    def __init__(self, logger ,config : WaveformConfig, injct_params_waves = None):
        self.logger             = logger
        self.config             = config
        if injct_params_waves is None:
            self.injct_params_waves = self.GetWaveFormParams()
            
        #parameters to be initialized during set_up:
        self.ifos     = None
        self.noise_td = None #for plotting
        self.wg       = None
        
    def setUpScenario(self):
        """
        generates the waveform and interferrometer data structures
        
        """
        self.logger.info("$$$ setting up scenario; generating overlapping waves")
        
        #load realistic background and PSD
        asd_f, asd = np.loadtxt(self.config.ASD_file_name+".txt", unpack=True)  
        psd        = asd**2
        np.savetxt(self.config.ASD_file_name+"_PSD"+".txt", np.column_stack([asd_f, psd]))
        
        et1       = bilby.gw.detector.get_empty_interferometer("L1")
        # et2       = bilby.gw.detector.get_empty_interferometer("ET2")
        # et3       = bilby.gw.detector.get_empty_interferometer("ET3")
        self.ifos = bilby.gw.detector.InterferometerList([et1]) #, et2, et3
        
        #load spectral density according to the file
        for ifo in self.ifos:
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
                psd_file=self.config.ASD_file_name+"_PSD"+".txt"
            )

        #add gaussian noise
        self.ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.config.sampling_frequency,
            duration=self.config.duration,
            start_time=0.0
        )
        
        #noise background used for plotting
        i = 0
        self.noise_td = []
        for ifo in self.ifos:
            self.noise_td.append(self.ifos[i].strain_data.time_domain_strain.copy())
            i += 1
            
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
            self.ifos.inject_signal(
                waveform_generator=self.wg,
                parameters=inject_params
            )
    
    def _getDataTimeSeries(self,strainI):
        """
        get timeseries objects of original noise and signal+noise for plotting 
        
        Args:
            sampling_frequency (_type_): _description_
        Returns:
            ts: original timeseries with noise + signal
            ts_noise: timeseries noise 
        """
        self.logger.info("$$$ getting time series objects")
        td = self.ifos[strainI].strain_data.time_domain_strain          # numpy array (length = duration * fs)
        t0 = self.ifos[strainI].strain_data.start_time                  # GPS start time (float)
        fs = self.config.sampling_frequency

        #convert data
        ts       = TimeSeries(td, sample_rate=fs, epoch=t0)
        ts_noise = TimeSeries(self.noise_td[strainI], dt=1/fs, epoch=t0)
        return ts, ts_noise
    
    ### plots ###
    def _PlotTimeSignal(self,ts,timeCenter,ts_noise,fileName):
        #plot time domain of signal
        self.logger.info("$$$ making time domain plot")
        white   = ts.whiten(4, 2).bandpass(40, 200)
        white_noise = ts_noise.whiten(4, 2).bandpass(40, 200)
        t_start = timeCenter - 2.0
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
            outseg=(timeCenter - 3, timeCenter + 3), 
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
        ts, ts_noise = self._getDataTimeSeries(0)
        tc           = self.injct_params_waves[0]['geocent_time']
        
        self._PlotTimeSignal(ts,tc,ts_noise,fileNames[0])
        self._PlotQtrans(tc,ts,fileNames[1])
    
    def GetWaveFormParams(self):
        self.logger.info("$$$ getting waveform parameters")
        
        #convert masses to chirp and ratio
        m1_1, m2_1   = 12.0, 10.0
        m1_2, m2_2   = 15.0, 10.0
        chirp_1, q_1 = massesToChirpAndQ(m1_1, m2_1)
        chirp_2, q_2 = massesToChirpAndQ(m1_2, m2_2)
        
        injct_params_wave_1 = dict(
            chirp_mass          = chirp_1,
            mass_ratio          = q_1,
            a_1                 = 0.6,  #part of the spin of the black hole
            a_2                 = 0.1,
            tilt_1              = 0.5, #part of the spin of the black hole
            tilt_2              = 2.3,
            phi_12              = 2.2,  #part of the spin of the black hole
            phi_jl              = 0.1,
            luminosity_distance = 3000.0, #2000
            theta_jn            = 1.2, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 1.9,
            geocent_time        = 3,
            ra                  = 1.375, #longituded
            dec                 = -1.2108,  #lattiude
        )
        injct_params_wave_2 = dict(
            chirp_mass          = chirp_2,
            mass_ratio          = q_2,
            a_1                 = 0.2,  #part of the spin of the black hole
            a_2                 = 0.9,
            tilt_1              = 0.2, #part of the spin of the black hole
            tilt_2              = 2.0,
            phi_12              = 5.7,  #part of the spin of the black hole
            phi_jl              = 1.3,
            luminosity_distance = 3000.0, #2000
            theta_jn            = 1.5, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 1.2,
            geocent_time        = 2.0,
            ra                  = 1.75, #longituded
            dec                 = -2.8,  #lattiude
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
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):

        self.run_sampler = run_sampler
        self.scenario    = scenario
        self.logger      = logger
        self.method_type = None
        self.config      = config
        
        #results
        self.posteriors = None
    
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
    
    def sampeler(self,resume):
        self.logger.info("$$$ Generating posterior samples using nested sampeling dynesty")
        clean = not resume
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
            resume     = resume,
            clean      = clean,
            outdir     = "out/outdir_ET_dynesty_" + self.method_type.code,
            label      = self.method_type.code,
            npool      = self.config.npool,
            queue_size = self.config.npool
        )
        return sample
    
    def generateSamples(self):
        self.logger.info("$$$ run the samples")
        #bayesian part
        if self.run_sampler:
            full_rerun = not self.run_sampler
            result = self.sampeler(full_rerun)
        else:
            outdir = "out/outdir_ET_dynesty_" + self.method_type.code + "/" + self.method_type.code + "_result.json"
            result = read_in_result(outdir) #outdir is also used in sampeler 
        self.posteriors = {"waveFormA" : result} #pandas data frame of samples
        
    ###   priors   ###
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
    
    @abstractmethod
    def getPrior(self):
        pass
    
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

class JointLikelihoodlMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):
        super().__init__(run_sampler, scenario, logger, config)
        self.method_type = Method_type.JOINT
        
    def likelihood(self):
        #adapt both prior and likelihood for joint modelling
        self.logger.info("$$$ get the joint likelihood signal")
        waveform_parms = self.scenario.GetWaveFormParams()
        ref_injection  = self.scenario.build_ref_injection(waveform_parms)
        
        wg_rb          = self._getRBWaveForm()
        likelihood     = OverlappingSignalsRelBinning(
            interferometers    = self.scenario.ifos,
            waveform_generator = wg_rb,
            ref_injection      = ref_injection, # actual parameters in simulation, ML for actual data, this is the FUDICIAL waveform used in the RB scheme
            N_overlaps         = 2,
            priors             = self.getPrior(),
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
    
    def getPrior(self):
        return self.getJointPriors()
    
class HyrarchicalMethod(Method):
    def __init__(self,run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig):
        super().__init__(run_sampler, scenario, logger, config)
        self.method_type     = Method_type.HIERARCHICAL
        self.singleSampler   = SingleSignalMethod(True, scenario, logger,config) #run_sampler to true so that we always generate a new sample instead of using the one from the single method
        self.singleSampler.nameExtra = "1" 
        self.second_wave_ifos = self.scenario.ifos
        
    def generateSamples(self):
        self.logger.info("$$$ generate Samples for hyrarchical model")
        # 
        self.singleSampler.generateSamples()
        
        posteriorSampleA     = self.singleSampler.posteriors['waveFormA']
        MLPosteriorA         = getMaximumLikelihood(posteriorSampleA)
        pols                 = self.scenario.wg.frequency_domain_strain(MLPosteriorA) #returns cross and plus waveform
        second_wave_ifos     = []
        for ifo in self.scenario.ifos:
            h_fd                 = ifo.get_detector_response(pols, MLPosteriorA)
            d_fd                 = ifo.strain_data.frequency_domain_strain
            res_fd               = d_fd - h_fd
            second_wave_ifo      = self._GetIfoResidual(res_fd,ifo)
            second_wave_ifos.append(second_wave_ifo)
        self.second_wave_ifos = InterferometerList(second_wave_ifos)
        super().generateSamples()
        posteriorSampleB     = self.singleSampler.posteriors['waveFormA']
        self.posteriors = {
            "waveFormA" : posteriorSampleA,
            "waveFormB" : posteriorSampleB
        }
        
        
    def likelihood(self):
        self.logger.info("$$$ get likelihood sgnal for custom ifo")
        prior = self.GetSinglePrior()
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers          = self.second_wave_ifos,
            waveform_generator       = self.scenario.wg,
            priors                   = prior,
            distance_marginalization = False,
            phase_marginalization    = False,
            time_marginalization     = False,
            # reference_frame="H1L1", #depends on the detector config -> ok?
            # time_reference="H1",
        )
        return likelihood

    def _GetIfoResidual(self,res_fd,ifo):
        self.logger.info("$$$ get residual interferrometer")
        #build copy for second interferrometer
        new_ifo = get_empty_interferometer(ifo.name)
        new_ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain = res_fd,
            sampling_frequency      = ifo.strain_data.sampling_frequency,
            duration                = ifo.strain_data.duration,
            start_time              = ifo.strain_data.start_time,
        )
        new_ifo.power_spectral_density = ifo.power_spectral_density
        return new_ifo
    
    def getPrior(self):
        return self.GetSinglePrior()
    

class Method_type(Enum):
    SINGLE       = ("single_likl", SingleSignalMethod)
    HIERARCHICAL = ("hierarchical",HyrarchicalMethod)  
    JOINT        = ("joint_likl", JointLikelihoodlMethod)  #this needs to be first

    
    def __init__(self, code, method: Method):
        self.code   = code
        self.method = method

### helper functions ###

def getMaximumLikelihood(result):
    posterior = result.posterior
    idx_ml    = posterior["log_likelihood"].idxmax()
    ml_sample = posterior.loc[idx_ml]
    return {k: ml_sample[k] for k in result.search_parameter_keys} #format for waveform generator
    
def massesToChirpAndQ(m1, m2):
    # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
    if m1 < m2:
        m1, m2 = m2, m1
    q = m2 / m1                       # mass_ratio in (0, 1]
    # chirp mass in solar masses
    chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
    return chirp, q

def SaveResults(results, out_dir="out", filename="results.json"):
    """Write a dictionary to a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path

def plotOverlap(params,results,truths,logger):
    logger.info("$$$ Making corner plots")
    fig                = None  #necessary for initialization
    params_A, params_B = addSuffixes(params)
    
    #colors
    cmap               = plt.get_cmap("tab20")
    colors             = list(cmap.colors)        # length 20
    n_colors           = len(colors)
    color_idx          = 0 
    legend_handles = []  
    legend_labels  = []
    
    for method in results:
        res = results[method]
        for waveform in res:
            wave = res[waveform]
            color = colors[color_idx % n_colors]
            color_idx += 1            #separate the joint poisterior that ends with _A and _B in their respective posterior samples
            label = f"{method.code} – {waveform}"
            
            if method == Method_type.JOINT:
                df_A = wave.posterior.copy()
                df_A.rename(columns=params_A, inplace=True)
                df_B = wave.posterior.copy()
                df_B.rename(columns=params_B, inplace=True)
                fig = createCornerPlot(df_A[params].values,params,color,fig,truths[1])  #index doesn't matter as things get overlapped
                fig = createCornerPlot(df_B[params].values,params,color,fig,truths[0])
            else:
                #Single waveform
                fig = createCornerPlot(wave.posterior[params].values,params,color,fig,truths[0])
            legend_handles.append(
                Line2D([0], [0], color=color, lw=2)
            )
            legend_labels.append(label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig("Plots/multiple waveforms.png", dpi=200)
    
def createCornerPlot(samps,params,color,fig,truths):
    fig = corner.corner(
        samps,
        labels        = params,          # base labels, no _A/_B
        color         = color,
        truths        = truths,
        truth_color   = "black",
        plot_contours = True,
        fill_contours = False,
        hist_kwargs   = dict(density=True),
        fig           = fig,    # None for first call; existing fig later
    )
    return fig

def addSuffixes(strings):
    suffixed_A = {s + "_A":s for s in strings}
    suffixed_B = {s + "_B":s for s in strings}
    return suffixed_A, suffixed_B


###########################
####     Main Loop     ####
###########################

def Main(run_sampler,saveResults):
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    # needed for multi threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    #Logger
    bilby.core.utils.setup_logger(
        log_level="INFO",
        label="my_run", 
        outdir="out",
    )
    logger = bilby.core.utils.logger
    logger.info("$$$ start_run")
    
    #build scenario
    ScenarioConfig = WaveformConfig()
    scenario       = GWScenario(logger, ScenarioConfig)
    scenario.setUpScenario()
    scenario.makePlots(["strain_time_domain_set_up","qtransform_set_up"])
    
    results       = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for measuring overlap etc with the joint.
    samplesToPlot = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for plotting
    MethodConf    = MethodConfig()
    
    JointBaselineDistr = None
    for method_type in Method_type:
        logger.info(f"$$$ Running method: {method_type.code}")
        method  = method_type.method(run_sampler,scenario,logger,MethodConf)
        
        start                           = time.process_time()
        method.generateSamples()
        end                             = time.process_time()
        runTime                         = end - start
        results[method_type.code]['runTime'] = runTime
        # if method_type == Method_type.JOINT:
        #     JointBaselineDistr
        # elif JointBaselineDistr is None:
        #     raise ValueError("joint serves as a baseline and needs to be ran first")
        
        # else: pass
        
        #plotting info
        sample                          = method.posteriors
        samplesToPlot[method_type]      = sample
        
        #build measurements
        diagnostics = {}
        diagnostics["information_gain"] = []
        for waveform in sample.keys():
            diagnostics["information_gain"].append(sample[waveform].information_gain)
        results[method_type.code]['diagnostics'] = diagnostics
        
        # analyze the samples
    
    #post processing
    # injct_params_wave = sample.to_dict(orient="records")[0:1]
    # SetupSignalAndDetector(wfv_args,ASD_file_name,injct_params_wave,True,logger)
    if saveResults:
        SaveResults(results)
        
    #make corner plot of the posterior samples
    params = ["chirp_mass", "mass_ratio", "luminosity_distance"] #,"mass_ratio","luminosity_distance"
    truths = scenario.GetWaveFormParams()
    truths = [[d[k] for k in params if k in d] for d in truths]  #get params to be plotted in corner plot
    plotOverlap(params,samplesToPlot,truths,logger)

###########################
###   Run actual Code   ###
###########################
Main(run_sampler = True, saveResults = True)