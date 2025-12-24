import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector.networks import TriangularInterferometer
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from dataclasses import asdict
from scenario.ScenarioConfig import ScenarioConfig
import os

class GWScenario:
    def __init__(self, logger ,config : ScenarioConfig, injct_params_waves = None):
        self.logger             = logger
        self.config             = config
        if injct_params_waves is None:
            self.injct_params_waves = self.GetWaveFormParams()
            
        #set-up plotting dirs separate from post processing dir
        #parameters to be initialized during set_up:
        self.ifos     = None
        self.noise_td = None #for plotting
        self.wg       = None
        
    def setUpScenario(self):
        """
        generates the waveform and interferrometer data structures
        
        """
        self.logger.info("$$$ setting up scenario; generating overlapping waves")
        
        PSD_CE = self._createPSDFile(self.config.ASD_file_name_CE)
        PSD_ET = self._createPSDFile(self.config.ASD_file_name_ET)
        self.ifos = self._getInterferrometerSetUp(PSD_ET,PSD_CE)
        
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
            waveform_arguments={k: v for k, v in asdict(self.config).items() if k in {"waveform_approximant","minimum_frequency","reference_frequency"
        }}  #remove redundant variables
        )
        
        #inject N waves
        self.logger.info("$$$ injecting " + str(len(self.injct_params_waves)) + " waves")
        for inject_params in self.injct_params_waves:
            self.ifos.inject_signal(
                waveform_generator=self.wg,
                parameters=inject_params
            )
    def _getInterferrometerSetUp(self,PSD_ET,PSD_CE):
        #einstein set-up at rhine meuse
        latitude_deg  = 50.85      
        longitude_deg = 5.70    
        elevation_m   = 100.0   #altitde of detector
        
        xarm_azimuth_deg = 0.0
        yarm_azimuth_deg = xarm_azimuth_deg + 60.0

        f_min = self.config.minimum_frequency
        f_max = self.config.sampling_frequency / 2
        ifos_1 = TriangularInterferometer(
            name="ET",
            minimum_frequency=f_min,   # choose consistently with your waveform and PSD validity
            maximum_frequency=f_max,                          # e.g. Nyquist-ish; bilby will also use your strain settings
            length=10.0,    #km                              
            latitude=latitude_deg,
            longitude=longitude_deg,
            elevation=elevation_m,
            power_spectral_density=PSD_ET,
            xarm_azimuth=xarm_azimuth_deg,
            yarm_azimuth=yarm_azimuth_deg
        )
        
        #CE set-up
        H1 = bilby.gw.detector.get_empty_interferometer("H1")
        L1 = bilby.gw.detector.get_empty_interferometer("L1")
        ifos_2 = bilby.gw.detector.InterferometerList([H1, L1])
        
        #load spectral density according to the file
        for ifo in ifos_2:
            ifo.power_spectral_density = PSD_CE
            ifo.length = 40.0 #km
            ifo.minimum_frequency = f_min
            ifo.maximum_frequency = f_max
        ifos = bilby.gw.detector.InterferometerList(ifos_1 + ifos_2)
        
        return ifos
    
    def _getDataTimeSeries(self,strainI,ifos):
        """
        get timeseries objects of original noise and signal+noise for plotting 
        
        Args:
            sampling_frequency (_type_): _description_
        Returns:
            ts: original timeseries with noise + signal
            ts_noise: timeseries noise 
        """
        #use default scenario based ifo's
        if ifos is None:
            ifos = self.ifos
            
        self.logger.info("$$$ getting time series objects")
        td = ifos[strainI].strain_data.time_domain_strain          # numpy array (length = duration * fs)
        t0 = ifos[strainI].strain_data.start_time                  # GPS start time (float)
        fs = self.config.sampling_frequency

        #convert data
        ts       = TimeSeries(td, sample_rate=fs, epoch=t0)
        ts_noise = TimeSeries(self.noise_td[strainI], dt=1/fs, epoch=t0)
        return ts, ts_noise
    
    ### plots ###
    def _PlotTimeSignal(self,ts,timeCenter,ts_noise,fileName,outDir):
        #plot time domain of signal
        self.logger.info("$$$ making time domain plot")
        white   = ts.whiten(4, 2).bandpass(40, 200)
        white_noise = ts_noise.whiten(4, 2).bandpass(40, 200)
        t_start = timeCenter - 3.0
        t_end   = timeCenter + 0.5

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
                color="red", alpha=0.2, lw=1.2, label="full signal")
        # Raw data (grey, semi-transparent)
        ax.plot(white_zoom.times, raw_scaled,
                color="black", alpha=1, lw=1, label="signal")
        # Styling
        ax.set_title("ET1 strain: time domain")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Strain")
        ax.legend(loc="upper right")
        
        path = os.path.join(outDir, f"{fileName}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
                
    def _PlotQtrans(self,timeCenter,ts,fileName,outDir):
        self.logger.info("$$$ making Qtransformed plot")
        qspec = ts.q_transform(
            qrange=(8, 8),
            frange=(20, 512),
            outseg=(timeCenter - self.config.duration//2, timeCenter + self.config.duration//2), 
            whiten=True,              
        )

        # Let GWpy handle the plotting
        fig = qspec.plot()
        ax = fig.gca()
        ax.set_yscale("log")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Q-transform of strain around first merger")
        path = os.path.join(outDir, f"{fileName}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        
    def makePlots(self,fileNames,outDir,ifos = None):
        self.logger.info("$$$ making plots")
        # data plots
        # corner plot already done
        ts, ts_noise = self._getDataTimeSeries(0,ifos)
        tc           = self.injct_params_waves[0]['geocent_time']
        
        self._PlotTimeSignal(ts,tc,ts_noise,fileNames[0],outDir)
        self._PlotQtrans(tc,ts,fileNames[1],outDir)
    
    def GetWaveFormParams(self):
        self.logger.info("$$$ getting waveform parameters")
        
        #convert masses to chirp and ratio
        m1_1, m2_1   = 12.0, 10.0
        m1_2, m2_2   = 15.0, 10.0
        chirp_1, q_1 = self._massesToChirpAndQ(m1_1, m2_1)
        chirp_2, q_2 = self._massesToChirpAndQ(m1_2, m2_2)
        
        injct_params_wave_1 = dict(
            chirp_mass          = chirp_1,
            mass_ratio          = q_1,
            a_1                 = 0.6,  #part of the spin of the black hole
            a_2                 = 0.1,
            tilt_1              = 0.5, #part of the spin of the black hole
            tilt_2              = 0.3,
            phi_12              = 0.2,  #part of the spin of the black hole
            phi_jl              = 0.1,
            luminosity_distance = 5000.0, #2000
            theta_jn            = 0.2, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 0.9,
            geocent_time        = self.config.duration/2,# 0.5,
            ra                  = 1.375, #longituded
            dec                 = -1.2108,  #lattiude
        )
        # injct_params_wave_2 = dict(
        #     chirp_mass          = chirp_2,
        #     mass_ratio          = q_2,
        #     a_1                 = 0.2,  #part of the spin of the black hole
        #     a_2                 = 0.9,
        #     tilt_1              = 0.2, #part of the spin of the black hole
        #     tilt_2              = 2.0,
        #     phi_12              = 5.7,  #part of the spin of the black hole
        #     phi_jl              = 1.3,
        #     luminosity_distance = 5000.0, #2000
        #     theta_jn            = 1.5, #angle of angular momentum
        #     psi                 = 2.659,  #angle of polarization
        #     phase               = 1.2,
        #     geocent_time        = self.config.duration/2 - time_delta,
        #     ra                  = 1.75, #longituded
        #     dec                 = -2.8,  #lattiude
        # )
        injct_params_waves = [injct_params_wave_1]#,injct_params_wave_2]
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
    def _massesToChirpAndQ(self,m1, m2):
        # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
        if m1 < m2:
            m1, m2 = m2, m1
        q = m2 / m1                       # mass_ratio in (0, 1]
        # chirp mass in solar masses
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return chirp, q
    
    def _createPSDFile(self,ASD_file_name):
        #load realistic background and PSD
        asd_f, asd = np.loadtxt(ASD_file_name+".txt", unpack=True)  
        psd        = asd**2
        np.savetxt(ASD_file_name+"_PSD"+".txt", np.column_stack([asd_f, psd]))
        PSD = bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file( 
            psd_file=ASD_file_name +"_PSD"+".txt"
            )
        return PSD