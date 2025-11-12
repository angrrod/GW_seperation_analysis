import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole
from gwpy.timeseries import TimeSeries

### this file consists of the main loop for the tests conscerning GW separation analysis ###
def GenerateOverlapScenario(wfv_args,injct_params_wave_1,injct_params_wave_2):
    ifo = bilby.gw.detector.get_empty_interferometer("L1") #change name?
    #load spectral density according to the file
    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
        psd_file="ET_B_PSD.txt"
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
        waveform_arguments=wfv_args
    )
    
    ifo.inject_signal(
    waveform_generator=wg,
    parameters=injct_params_wave_1
    )

    ifo.inject_signal(
        waveform_generator=wg,
        parameters=injct_params_wave_2
    )

    td = ifo.strain_data.time_domain_strain          # numpy array (length = duration * fs)
    t0 = ifo.strain_data.start_time                  # GPS start time (float)
    fs = wfv_args['sampling_frequency']

    #convert data
    ts = TimeSeries(td, sample_rate=fs, epoch=t0)
    return ts, ifo

def Main():
    wfv_args = dict(
        waveform_approximant="IMRPhenomPv2", #IMRPhenomPv2 
        minimum_frequency=10.0,
        sampling_frequency=4096.0,
        reference_frequency=10.0,
        duration = 10,
    )
    injct_params_wave_1 = dict(
        mass_1=36.0,
        mass_2=29.0,
        a_1=0.4,  #part of the spin of the black hole
        a_2=0.3,
        tilt_1=0.5, #part of the spin of the black hole
        tilt_2=1.0,
        phi_12=1.7,  #part of the spin of the black hole
        phi_jl=0.3,
        luminosity_distance=100.0, #2000
        theta_jn=0.4, #angle of angular momentum
        psi=2.659,  #angle of polarization
        phase=1.3,
        geocent_time=5,
        ra=1.375, #longituded
        dec=-1.2108,  #lattiude
    )
    injct_params_wave_2 = dict(
        mass_1=34.0,
        mass_2=40.0,
        a_1=0.4,  #part of the spin of the black hole
        a_2=0.9,
        tilt_1=0.2, #part of the spin of the black hole
        tilt_2=1.0,
        phi_12=5.7,  #part of the spin of the black hole
        phi_jl=0.3,
        luminosity_distance=100.0, #2000
        theta_jn=0.4, #angle of angular momentum
        psi=2.659,  #angle of polarization
        phase=1.2,
        geocent_time=4.5,
        ra=1.75, #longituded
        dec=-1.8,  #lattiude
    )
    
    #load realistic background and PSD
    asd_f, asd = np.loadtxt("ET_B.txt", unpack=True)  
    psd = asd**2
    np.savetxt("ET_B_PSD.txt", np.column_stack([asd_f, psd]))
    ts, ifo = GenerateOverlapScenario(wfv_args,injct_params_wave_1,injct_params_wave_2)

