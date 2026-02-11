import bilby
import numpy as np
from bilby.gw.source import lal_binary_black_hole
from bilby.gw.detector.networks import TriangularInterferometer
from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
from dataclasses import asdict
from scenario.ScenarioConfig import ScenarioConfig
import os, json, datetime
from pathlib import Path
import h5py
from dataclasses import asdict
from utils import get_base_log_dir
class GWScenario:
    def __init__(self, logger ,scenarioId, config : ScenarioConfig, injct_params_waves = None):
        self.logger             = logger
        self.config             = config
        self.workdir            = get_base_log_dir() / "scenarioData"
        self.scenarioId         = scenarioId
        if injct_params_waves is None:
            self.injct_params_waves = self.GetWaveFormParams()
        else:
            self.injct_params_waves = injct_params_waves
            
        #set-up plotting dirs separate from post processing dir
        #parameters to be initialized during set_up:
        self.ifos          = None
        self.noise_td      = None #for plotting
        self.wg            = None
        
    def setUpScenario(self):
        """
        generates the waveform and interferrometer data structures
        
        """
        self.logger.info("$$$ setting up scenario; generating overlapping waves")
        
        PSD_CE = self.load_psd(self.config.ASD_file_name_CE +"_PSD.txt")
        PSD_ET = self.load_psd(self.config.ASD_file_name_ET +"_PSD.txt")
        
        self.ifos = self._getInterferrometerSetUp(PSD_ET,PSD_CE)
        
        #load gaussian noise
        if self.getScenarioPath().exists():
            self.loadScenario()
        else:
            # new scenario create gaussian noise
            self.ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency = self.config.sampling_frequency,
                duration           = self.config.duration,
                start_time         = self.config.start_time
            )
            self.saveScenario()
        
        #noise background used for plotting
        self.noise_td = []
        for ifo in self.ifos:
            self.noise_td.append(ifo.strain_data.time_domain_strain.copy())
            
        # BBH signal
        # waveform generator for normal likelihood model
        self.wg = bilby.gw.waveform_generator.WaveformGenerator(
            duration                      = self.config.duration,
            sampling_frequency            = self.config.sampling_frequency,
            frequency_domain_source_model = lal_binary_black_hole,
            parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments            = {k: v for k, v in asdict(self.config).items() if k in {"waveform_approximant","minimum_frequency","reference_frequency"
        }}  
        )
        
        #inject N waves
        self.logger.info("$$$ injecting " + str(len(self.injct_params_waves)) + " waves")
        for inject_params in self.injct_params_waves:
            self.ifos.inject_signal(
                waveform_generator=self.wg,
                parameters=inject_params
            )
        converted = self.wg.parameter_conversion(inject_params.copy())
        for k in [
            "mass_1", "mass_2",
            "chirp_mass", "mass_ratio",
            "luminosity_distance",
            "theta_jn", "inclination",
            "phase", "coa_phase",
            "geocent_time"
            ]: 
            if k in converted[0]:
                self.logger.info(f"$$$ {k}, {converted[0].get(k)}")
                
    def _getInterferrometerSetUp(self,PSD_ET,PSD_CE):
        
        # TODO: Test this
        #ifos = bilby.gw.detector.InterferometerList(['CE', 'ET'])

        #Einstein telescope set-up at rhine meuse
        latitude_deg  = 50.85      
        longitude_deg = 5.70    
        elevation_m   = 100.0   #altitude of detector
        
        xarm_azimuth_deg = 0.0
        yarm_azimuth_deg = xarm_azimuth_deg + 60.0

        f_min  = self.config.minimum_frequency
        f_max  = self.config.sampling_frequency / 2
        ifos_1 = TriangularInterferometer(
            name                   = "ET",
            minimum_frequency      = f_min,   # choose consistently with your waveform and PSD validity
            maximum_frequency      = f_max,   # e.g. Nyquist-ish; bilby will also use your strain settings
            length                 = 10.0,               # km                              
            latitude               = latitude_deg,
            longitude              = longitude_deg,
            elevation              = elevation_m,
            power_spectral_density = PSD_ET,
            xarm_azimuth           = xarm_azimuth_deg,
            yarm_azimuth           = yarm_azimuth_deg
        )
        
        # CE set-up
        H1     = bilby.gw.detector.get_empty_interferometer("H1")
        L1     = bilby.gw.detector.get_empty_interferometer("L1")
        ifos_2 = bilby.gw.detector.InterferometerList([H1, L1])
        
        # Load spectral density according to the file
        for ifo in ifos_2:
            ifo.power_spectral_density = PSD_CE
            ifo.length                 = 40.0 #km
            ifo.minimum_frequency      = f_min
            ifo.maximum_frequency      = f_max
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
        # Use default scenario based ifo's
        if ifos is None:
            ifos = self.ifos
            
        self.logger.info("$$$ getting time series objects")
        td = ifos[strainI].strain_data.time_domain_strain          # numpy array (length = duration * fs)
        t0 = ifos[strainI].strain_data.start_time                  # GPS start time (float)
        fs = self.config.sampling_frequency

        # Convert data
        ts       = TimeSeries(td, sample_rate=fs, epoch=t0)
        ts_noise = TimeSeries(self.noise_td[strainI], dt=1/fs, epoch=t0)
        return ts, ts_noise
    
    ### plots ###
    def _PlotTimeSignal(self,ts,timeCenter,ts_noise,fileName,outDir):
        #plot time domain of signal
        self.logger.info("$$$ making time domain plot")
        white       = ts.whiten(4, 2).bandpass(40, 200)
        white_noise = ts_noise.whiten(4, 2).bandpass(40, 200)
        t_start     = timeCenter - 5.0
        t_end       = timeCenter + 0.5

        white_zoom       = white.crop(t_start, t_end)
        white_noise_zoom = white_noise.crop(t_start, t_end)
        
        #rescale
        pure_signal = white_zoom.value - white_noise_zoom.value
        white       = white_zoom.value

        raw_max   = np.max(np.abs(pure_signal))
        white_max = np.max(np.abs(white))

        scale      = white_max / raw_max        # bring raw up to whitened level
        raw_scaled = pure_signal * scale
        
        # Plot manually using matplotlib (allows full control)
        fig, ax = plt.subplots(figsize=(8, 4))

        # Whitened signal (red, more prominent)
        ax.plot(white_zoom.times, white_zoom.value,
                color="red", alpha=0.4, lw=1.2, label="full signal")
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
            outseg=(0, self.config.duration), 
            whiten=True,              
        )

        # Let GWpy handle the plotting
        fig  = qspec.plot()
        ax   = fig.gca()
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
        m1_1, m2_1   = 10.0, 8.0
        m1_2, m2_2   = 15.0, 10.0
        chirp_1, q_1 = self._massesToChirpAndQ(m1_1, m2_1)
        chirp_2, q_2 = self._massesToChirpAndQ(m1_2, m2_2)
        
        self.logger.info(f"$$$ chirpmass {chirp_1} and mass ratio {q_1} for waveform 1")
        self.logger.info(f"$$$ chirpmass {chirp_2} and mass ratio {q_2} for waveform 1")
        
        injct_params_wave_1 = dict(
            chirp_mass          = chirp_1,
            mass_ratio          = q_1,
            a_1                 = 0.0,  #part of the spin of the black hole
            a_2                 = 0.0,
            tilt_1              = 0.0, #part of the spin of the black hole
            tilt_2              = 0.0,
            phi_12              = 0.0,  #part of the spin of the black hole
            phi_jl              = 0.0,
            luminosity_distance = 3000.0, #2000
            theta_jn            = 0.2, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 0.9,
            geocent_time        = self.config.duration*0.8,# 0.5,
            ra                  = 1.375, #longituded
            dec                 = -1.2108,  #lattiude
        )
        injct_params_wave_2 = dict(
            chirp_mass          = chirp_2,
            mass_ratio          = q_2,
            a_1                 = 0.0,  #part of the spin of the black hole
            a_2                 = 0.0,
            tilt_1              = 0.0, #part of the spin of the black hole
            tilt_2              = 0.0,
            phi_12              = 0.0,  #part of the spin of the black hole
            phi_jl              = 0.0,
            luminosity_distance = 5000.0, #2000
            theta_jn            = 1.5, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 1.2,
            geocent_time        = self.config.duration*0.8 - self.config.time_delta,
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
    
    def _massesToChirpAndQ(self,m1, m2):
        # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
        if m1 < m2:
            m1, m2 = m2, m1
        q = m2 / m1                       # mass_ratio in (0, 1]
        # chirp mass in solar masses
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return chirp, q
    
    def load_psd(self,psd_name):
        # 1) Try HPC path
        base_path = os.environ.get("GW_INP_DIR")
        if base_path is None:
            base_path = "" 
        hpc_path  = os.path.join(base_path, psd_name)
        try:
            return bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
                psd_file=hpc_path
            )
        except FileNotFoundError:
            # 2) Fall back to local directory
            return bilby.gw.detector.PowerSpectralDensity.from_power_spectral_density_file(
                psd_file=psd_name
            )

    def getScenarioPath(self) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir / f"Scenario_{self.scenarioId}.h5"
        
    def saveScenario(self):
        # save the ifos set-up to deal with good reproducability 
        dir = self.getScenarioPath()
        self.logger.info(f"$$$ Save scenario to {dir}")
        
        meta_config = asdict(self.config)
        meta_inj = getattr(self, "injct_params_waves", None)
        with h5py.File(str(dir), "w") as f:
            
            # general meta data currently not used
            f.create_dataset("/meta/created_utc", data=np.bytes_(
                datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
            ))
            f.create_dataset("/meta/scenario_config_json",
                            data=np.bytes_(json.dumps(meta_config)))
            f.create_dataset("/meta/injections_json",
                            data=np.bytes_(json.dumps(meta_inj)))
            
            # construct the strain data
            g_ifos = f.require_group("ifos")
            for ifo in self.ifos:
                g = g_ifos.require_group(ifo.name)
                td = np.asarray(ifo.strain_data.time_domain_strain, dtype=np.float64) #translate to numpy
                if "strain_td" in g: #replace if it already esists
                    del g["strain_td"]
                g.create_dataset("strain_td", data=td, compression="gzip", compression_opts=4)
                
        
    def loadScenario(self):
        """
        Load ONLY the realized strain (time-domain) for the currently-constructed self.ifos
        from an HDF5 file and overwrite each IFO's strain_data accordingly.

        Assumes:
        - self.ifos is already created (e.g., via _getInterferrometerSetUp(...))
        - HDF5 layout: /ifos/<IFO_NAME>/strain_td and scalar metadata
        """
        dir = self.getScenarioPath()
        self.logger.info(f"$$$ Loaded scenario from {dir}")
        if not dir.exists():
            raise FileNotFoundError(dir)
        if self.ifos is None:
            raise RuntimeError("self.ifos is not initialized. Build interferometers before loading strain.")
        
        with h5py.File(str(dir), "r") as f:
            if "ifos" not in f:
                raise KeyError(f"Group '/ifos' missing in {dir}")

            g_ifos = f["ifos"]

            for ifo in self.ifos:
                name = ifo.name
                h5_group_path = f"ifos/{name}"
                if name not in g_ifos:
                    raise KeyError(f"IFO group '/{h5_group_path}' missing in {dir}")

                g = g_ifos[name]
                if "strain_td" not in g:
                    raise KeyError(f"Dataset '/{h5_group_path}/strain_td' missing in {dir}")

                ifo.strain_data.set_from_time_domain_strain(
                    time_domain_strain = np.asarray(g["strain_td"], dtype=np.float64),
                    sampling_frequency = self.config.sampling_frequency,
                    duration           = self.config.duration,
                    start_time         = self.config.start_time,
                )
