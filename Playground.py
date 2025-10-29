#################################
#####  codes to run in cli  #####
#################################

#####  run: conda activate gw  #####
#####  conda install -c conda-forge lalsuite bilby gwpy matplotlib  #####

###################################################################################################

import gwpy  #visualization
import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from bilby.gw.source import lal_binary_black_hole

#TODO: test for multiple detectors
#TODO: ROQ is not good for E3

wfv_args = dict(
    waveform_approximant="IMRPhenomD", #IMRPhenomPv2 
    minimum_frequency=10.0,
    sampling_frequency=4096.0,
    reference_frequency=10.0,
    duration = 10,
)

ifo = bilby.gw.detector.get_empty_interferometer("L1") #change name?

#convert asd to psd
asd_f, asd = np.loadtxt("ET_B.txt", unpack=True)  
psd = asd**2
np.savetxt("ET_B_PSD.txt", np.column_stack([asd_f, psd]))

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
    frequency_domain_source_model=bilby.gw.source.binary_black_hole_roq,
    waveform_arguments=wfv_args
)

injct_params = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,  #part of the spin of the black hole
    a_2=0.3,
    tilt_1=0.5, #part of the spin of the black hole
    tilt_2=1.0,
    phi_12=1.7,  #part of the spin of the black hole
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4, #angle of angular momentum
    psi=2.659,  #angle of polarization
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375, #longituded
    dec=-1.2108,  #lattiude
)

ifo.inject_signal(
    waveform_generator=wg,
    parameters=injct_params
)

td = ifo.strain_data.time_domain_strain          # numpy array (length = duration * fs)
t0 = ifo.strain_data.start_time                  # GPS start time (float)
fs = wfv_args['sampling_frequency']

#convert data
ts = TimeSeries(td, sample_rate=fs, epoch=t0)

#plot data
fig1 = ts.plot()
ax = fig1.gca()
ax.set_title("ET1 strain: time domain")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Strain")
plt.show()

asd = ts.asd(fftlength=4, method="median")  # choose fftlength to balance resolution vs variance
fig2 = asd.plot()
ax = fig2.gca()
ax.set_xlim(5, fs/2)       # match your likelihood band and Nyquist
ax.set_ylim(1e-25, 1e-20)  # tweak to taste
ax.set_title("ET1 strain: amplitude spectral density")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("ASD [1/√Hz]")
plt.show()
