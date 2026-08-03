import bilby
import copy
from bilby.core.prior import DeltaFunction

class prior():
    def __init__(self,logger,MethodConfig,scenarioConfig):
        self.logger         = logger
        self.MethodConfig   = MethodConfig
        self.scenarioConfig = scenarioConfig
        
    def GetSinglePrior(self,waveformIdx = 0):
        self.logger.info("$$$ getting a waveform prior")
        prior = bilby.gw.prior.BBHPriorDict()  #allow for default ranges in ET
        if "geocent_time" not in prior:
            self.logger.info("$$$ geocent_time not in default prior, adding manually")
            prior["geocent_time"] = bilby.core.prior.Uniform(
                minimum=0, # maybe make this a bit bigger?
                maximum=self.scenarioConfig.duration,  
                name="geocent_time",
            )
        prior["chirp_mass"] = bilby.core.prior.Uniform(
            minimum=4, maximum=25, name="chirp_mass"
        )
        prior["mass_ratio"] = bilby.core.prior.Uniform(
            minimum=0.1, maximum=1, name="mass_ratio" 
        )
        prior["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            minimum=2e4,      
            maximum=1.5e5, #1e5      
            cosmology='Planck15',
            name='luminosity_distance',
            latex_label='$d_L$',
            unit='Mpc'
        )
        # fixed_priors   = ["geocent_time"]# "chirp_mass", ,"mass_ratio","luminosity_distance"
        fixed_priors = [
            # Spins and spin orientations: source A
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
            "phi_jl",
            "luminosity_distance",
            "dec",
            "ra",
            "theta_jn",
            "psi",
            "phase",
            "geocent_time",
            "delta_t_AB",
        ]
        prior = self.fixPriors(prior,fixed_priors,waveformIdx,self.MethodConfig.use_deltas)
        
        #fix priors for waveform
        if self.scenarioConfig.waveform_approximant == "IMRPhenomD":
            fixed_priors   = ["a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
            prior = self.fixPriors(prior,fixed_priors,waveformIdx,True)
        return prior
    
    ### WJ: 04/01/25 fix priors for debugging
    def fixPriors(self,prior,fixed_priors,waveformIdx,useDeltaFunction):
        waveFormParams = self.GetWaveFormParamsFixed()
        for k in fixed_priors:
            if k in prior: 
                value = waveFormParams[waveformIdx].get(k)
                if useDeltaFunction:
                    self.logger.info(f"$$$ making prior delta {k}")
                    prior[k] = DeltaFunction(value, name=k)
                elif self.MethodConfig.restrict_prior:
                    if (value > prior[k].maximum) or (value < prior[k].minimum):
                        raise ValueError(f'value: {value} not supported in prior [{prior[k].minimum},{prior[k].maximum}]')
                    upperDiff  = prior[k].maximum - value
                    lowerDiff  = value - prior[k].minimum
                    priorRange = (prior[k].maximum - prior[k].minimum)
                    tolerance  = 1e-3
                    delta      = self.MethodConfig.restriction_str * min(upperDiff,lowerDiff)
                    #deal with boundaries
                    if upperDiff < tolerance:
                        priorMin = value - priorRange*self.MethodConfig.restriction_str
                        priorMax = value + delta
                    elif lowerDiff < tolerance:
                        priorMin = value - delta
                        priorMax = value + priorRange*self.MethodConfig.restriction_str
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
        
        # WJ: 04/05/2026 Old code to break symmetry for injected signals
        # break prior symmetry, ensure that one signal is later than the other
        # max_time_A = self.GetWaveFormParamsFixed()[0].get("geocent_time") #used in symmetry breaking
        # max_time_B = self.GetWaveFormParamsFixed()[1].get("geocent_time")
        # if max_time_A > max_time_B:
        #     priors["geocent_time_B"] = bilby.core.prior.Uniform(
        #                                 minimum=40,
        #                                 maximum=max_time_A,  
        #                                 name="geocent_time_B",
        #                             )
        # else: 
        #     priors["geocent_time_A"] = bilby.core.prior.Uniform(
        #                                 minimum=40,
        #                                 maximum=max_time_B,  
        #                                 name="geocent_time_A",
        #                             )
        
        priors["geocent_time_A"] = bilby.core.prior.Uniform(
            minimum=20,
            maximum=50,
            name="geocent_time_A"
        )

        priors["delta_t_AB"] = bilby.core.prior.Uniform(
            minimum=1e-4,
            maximum=10,   # choose physically reasonable max separation
            name="delta_t"
        )
        del priors["geocent_time_B"]
        return priors
    
    def GetWaveFormParamsSampled(self):
        prior = self.getJointPriors()
        injct = prior.sample()
        dict_A = self.filter_dict_by_suffix(injct, "_A")
        dict_B = self.filter_dict_by_suffix(injct, "_B")
        # Reconstruct B time from A time and delta.
        # Your fixed setup has A later than B:
        # geocent_time_B = geocent_time_A - delta_t_AB
        if "geocent_time" not in dict_B:
            dict_B["geocent_time"] = dict_A["geocent_time"] - injct["delta_t_AB"]
        return [dict_A,dict_B]
    
    def filter_dict_by_suffix(self,d: dict, suffix: str) -> dict:
        """
        Return a new dictionary containing only keys that end with `suffix`.
        The suffix is removed from the returned keys.

        Example:
            {"x_A": 1, "y_B": 2}, suffix="_A"
            -> {"x": 1}
        """
        suffix_len = len(suffix)

        return {
            key[:-suffix_len]: value
            for key, value in d.items()
            if key.endswith(suffix)
        }
        
    def GetWaveFormParamsFixed(self):
        self.logger.info("$$$ getting fixed waveform parameters")
        
        #convert masses to chirp and ratio
        m1_1, m2_1   = 10.0, 8.0
        m1_2, m2_2   = 15.0, 10.0
        chirp_1, q_1 = self._massesToChirpAndQ(m1_1, m2_1)
        chirp_2, q_2 = self._massesToChirpAndQ(m1_2, m2_2)
        
        self.logger.info(f"$$$ fixed waveform params: chirpmass {chirp_1} and mass ratio {q_1} for waveform 1")
        self.logger.info(f"$$$ fixed waveform params: chirpmass {chirp_2} and mass ratio {q_2} for waveform 1")
        
        injct_params_wave_1 = dict(
            chirp_mass          = chirp_1,
            mass_ratio          = q_1,
            a_1                 = 0.0,  #part of the spin of the black hole
            a_2                 = 0.0,
            tilt_1              = 0.0, #part of the spin of the black hole
            tilt_2              = 0.0,
            phi_12              = 0.0,  #part of the spin of the black hole
            phi_jl              = 0.0,
            luminosity_distance = 9000.0, #2000
            theta_jn            = 0.2, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 0.9,
            geocent_time        = self.scenarioConfig.duration*0.8,# 0.5,
            ra                  = 1.375, 
            dec                 = -0.2108, 
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
            luminosity_distance = 8000.0, #2000
            theta_jn            = 1.5, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 1.2,
            geocent_time        = self.scenarioConfig.duration*0.8 - self.scenarioConfig.time_delta,
            ra                  = 1.2,
            dec                 = -1.2, 
        )
        injct_params_waves = [injct_params_wave_1,injct_params_wave_2]
        return injct_params_waves
    
    def _massesToChirpAndQ(self,m1, m2):
        # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
        if m1 < m2:
            m1, m2 = m2, m1
        q = m2 / m1                       # mass_ratio in (0, 1]
        # chirp mass in solar masses
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return chirp, q