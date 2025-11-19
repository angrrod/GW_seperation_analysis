import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from bilby.core.result import read_in_result

### this file consists of the main loop for the tests conscerning GW separation analysis ###
def GetScenario(wfv_args,ASD_file_name,logger):
    """generates the waveform and interferrometer data structures

    Args:
        wfv_args (ditc): arguments used in waveform generation
        injct_params_waves (list of signal dicts): list of all signals
        logger (bilby.core.utils.logger): used logger

    Returns:
        _type_: _description_
    """
    logger.info("generating overlapping waves")
    ifo = bilby.gw.detector.get_empty_interferometer("L1") #change name?
    #load spectral density according to the file
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
        psd_file=ASD_file_name+"_PSD"+".txt"
    )

    #add gaussian noise
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=wfv_args['sampling_frequency'],
        duration=wfv_args['duration'],
        start_time=0.0
    )

    # BBH signal
    wg = bilby.gw.waveform_generator.WaveformGenerator(
        duration=wfv_args['duration'],
        sampling_frequency=wfv_args['sampling_frequency'],
        frequency_domain_source_model=lal_binary_black_hole,
        waveform_arguments={k: v for k, v in wfv_args.items() if k not in {"sampling_frequency","duration"}}  #remove redundant variables
    )
    return ifo, wg

def InjectSignal(ifo, injct_params_waves,wg,logger):
    #add N waves
    logger.info("injecting " + str(len(injct_params_waves)) + " waves")
    for inject_params in injct_params_waves:
        ifo.inject_signal(
            waveform_generator=wg,
            parameters=inject_params
        )
    return ifo

def getDataTimeseries(ifo, sampling_frequency, logger):
    logger.info("getting time series object")
    td = ifo.strain_data.time_domain_strain          # numpy array (length = duration * fs)
    t0 = ifo.strain_data.start_time                  # GPS start time (float)
    fs = sampling_frequency

    #convert data
    ts = TimeSeries(td, sample_rate=fs, epoch=t0)
    return ts

def GetPriors(logger):
    logger.info("getting the priors")

    priors = bilby.gw.prior.BBHPriorDict()  #allow for default ranges in ET
    return priors

def GetSingleLikelihood(priors,ifos,waveform_generator,logger):
    logger.info("get a single likelihood signal")

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        distance_marginalization=True,
        phase_marginalization=True,
        time_marginalization=True,
        # reference_frame="H1L1", #depends on the detector config -> ok?
        # time_reference="H1",
    )
    return likelihood

def GeneratePosteriorSample(likelihood, priors,logger):
    logger.info("Generating posterior samples using nested sampeling dynesty")
    sample = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=50, 
        dlogz=2.0, #stopping criterion for the evidence
        sample="rwalk",  
        walks=10, #steps for MCMC sampeler to select new candidates     
        nact=3, #amount of steps is tuned so autocorr is small enough 
        resume=True,
        outdir="outdir_ET_dynesty",
        label="ET_BBH_example",
    )
    return sample

def GetWaveFormParams(logger):
    logger.info("getting waveform parameters")
    injct_params_wave_1 = dict(
        mass_1=12.0,
        mass_2=18.0,
        a_1=0.4,  #part of the spin of the black hole
        a_2=0.3,
        tilt_1=0.5, #part of the spin of the black hole
        tilt_2=1.0,
        phi_12=1.7,  #part of the spin of the black hole
        phi_jl=0.3,
        luminosity_distance=50000.0, #2000
        theta_jn=1.4, #angle of angular momentum
        psi=2.659,  #angle of polarization
        phase=1.3,
        geocent_time=5,
        ra=1.375, #longituded
        dec=-1.2108,  #lattiude
    )
    injct_params_wave_2 = dict(
        mass_1=21.0,
        mass_2=10.0,
        a_1=0.4,  #part of the spin of the black hole
        a_2=0.9,
        tilt_1=0.2, #part of the spin of the black hole
        tilt_2=1.0,
        phi_12=5.7,  #part of the spin of the black hole
        phi_jl=0.3,
        luminosity_distance=50000.0, #2000
        theta_jn=1.5, #angle of angular momentum
        psi=2.659,  #angle of polarization
        phase=1.2,
        geocent_time=4.5,
        ra=1.75, #longituded
        dec=-1.8,  #lattiude
    )
    injct_params_waves = [injct_params_wave_1,injct_params_wave_2]
    return injct_params_waves

def JPE():
    pass

def HyrarchicalEstimation():
    pass

### plots ###

def PlotTimeSignal(ts,logger):
    #plot time domain of signal
    logger.info("making time domain plot")
    fig1 = ts.plot()
    ax = fig1.gca()
    ax.set_title("ET1 strain: time domain")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Strain")
    fig1.savefig("Plots/strain_time_domain.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    
# def PlotASD(fs,ts,logger):
#     logger.info("making ASD plot")
#     #plot ASD
#     asd = ts.asd(fftlength=4, method="median")  # choose fftlength to balance resolution vs variance
#     fig2 = asd.plot()
#     ax = fig2.gca()
#     ax.set_xlim(5, fs/2)       # match your likelihood band and Nyquist
#     ax.set_ylim(1e-25, 1e-20)  # tweak to taste
#     ax.set_title("ET1 strain: amplitude spectral density")
#     ax.set_xlabel("Frequency [Hz]")
#     ax.set_ylabel("ASD [1/√Hz]")
#     fig2.savefig("Plots/strain_asd.png", dpi=300, bbox_inches="tight")
#     plt.close(fig2)

def PlotQtrans(tc,ts,logger):
    logger.info("making Qtransformed plot")
    qspec = ts.q_transform(
        qrange=(8, 8),
        frange=(20, 512),
        outseg=(tc - 4, tc + 4),  # time window around tc
        whiten=True,              # default in many versions, but explicit is fine
    )

    # Let GWpy handle the plotting
    fig = qspec.plot()
    ax = fig.gca()
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Q-transform of strain around first merger")
    fig.savefig("Plots/qtransform.png", dpi=300, bbox_inches="tight")

def Main(run_sampler):
    
    #Logger
    bilby.core.utils.setup_logger(
        log_level="INFO",
        label="my_run", 
        outdir="out",
    )
    logger = bilby.core.utils.logger
    logger.info("start_run")
    
    wfv_args = dict(
        waveform_approximant="IMRPhenomPv2", #IMRPhenomPv2 
        minimum_frequency=10.0,
        sampling_frequency=4096.0,
        reference_frequency=10.0,
        duration = 10,
    )
    
    injct_params_waves = GetWaveFormParams(logger)
    
    #load realistic background and PSD
    ASD_file_name = "ET_C"
    asd_f, asd = np.loadtxt(ASD_file_name+".txt", unpack=True)  
    psd        = asd**2
    np.savetxt(ASD_file_name+"_PSD"+".txt", np.column_stack([asd_f, psd]))
    ifo, wg    = GetScenario(wfv_args,ASD_file_name,logger)
    ifo        = InjectSignal(ifo, injct_params_waves,wg,logger)
    ifos       = [ifo] #list of interferrometers
    
    # data plots
    #corner plot already done
    ts = getDataTimeseries(ifo,wfv_args["sampling_frequency"],logger)
    PlotTimeSignal(ts, logger)
    
    tc = injct_params_waves[0]['geocent_time']
    PlotQtrans(tc,ts,logger)
    
    # analysis
    priors     = GetPriors(logger)
    likelihood = GetSingleLikelihood(priors,ifos,wg,logger)
    
    if run_sampler:
        result = GeneratePosteriorSample(likelihood, priors ,logger)
    else:
        result = read_in_result("outdir_ET_dynesty/ET_BBH_example_result.json")
    sample = result.posterior
    
    #make corner plot of the posterior samples
    result.plot_corner()

###########################
###   Run actual Code   ###
###########################

Main(run_sampler = True)