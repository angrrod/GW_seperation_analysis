import bilby
import copy
from bilby.core.prior import DeltaFunction
from utils import load_config

class SamplerPrior():
    def __init__(self,logger):
        self.logger  = logger
        
        self.configs = load_config()
        self.intrinsic_prior_config = load_config(
            "waveform_dataset_settings.yml"
        )["intrinsic_prior"]

        self.extrinsic_prior_config = load_config(
            "training_copula.yml"
        )["data"]["extrinsic_prior"]
        self.prior_config = {
        **self.intrinsic_prior_config,
        **self.extrinsic_prior_config,
    }


    def _build_single_prior(self):
        single_prior = bilby.core.prior.PriorDict()
        joint_prior = self._build_joint_prior()
        suffix = "_A"

        for key, parameter_prior in joint_prior.items():
            if not key.endswith(suffix):
                continue

            new_key = key[:-len(suffix)]

            parameter_prior = copy.deepcopy(parameter_prior)

            # Keep the Prior object's internal name consistent with its new key.
            if hasattr(parameter_prior, "name"):
                parameter_prior.name = new_key

            single_prior[new_key] = parameter_prior
        return single_prior
    
    def GetSinglePrior(self,waveformIdx = 0):
        self.logger.info("$$$ getting a waveform prior")
        single_prior = self._build_single_prior()

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
        ]
        #doesn't fix if 'use_deltas' isn't switched to true
        prior = self.fixPriors(single_prior,fixed_priors,waveformIdx,self.configs["prior_controls"]["use_deltas"])
        
        #fix priors for waveform
        if self.configs["waveform_generator"]["approximant"] == "IMRPhenomD":
            fixed_priors   = ["a_1","a_2","tilt_1","tilt_2","phi_12","phi_jl"]
            prior = self.fixPriors(single_prior,fixed_priors,waveformIdx,True)
        return prior
    
    ### WJ: 04/01/25 fix priors for debugging
    def fixPriors(self,priors,fixed_priors,waveformIdx,useDeltaFunction):
        waveform_params = self.GetWaveFormParamsFixed()
        params = waveform_params[waveformIdx]

        for key in fixed_priors:
            if key not in priors:
                continue

            value = params.get(key)

            if value is None:
                continue

            priors = self._fixPriorValue(
                priors,
                key,
                value,
                useDeltaFunction,
            )

        return priors
    
    # needed for joint parameter estimation
    # independent priors for both
    def getJointPriors(self):
        self.logger.info("$$$ getting joint priors")
        #get waveforms with prior around actual values
        priors = self._build_joint_prior()
        # Apply the normal scenario prior controls.
        priors = self.fixJointPriors(
            priors,
            self.configs["prior_controls"]["use_deltas"],
        )
        
        # IMRPhenomD requires non-precessing/fixed spin parameters.
        if self.configs["waveform_generator"]["approximant"] == "IMRPhenomD":
            spin_parameters = [
                "a_1",
                "a_2",
                "tilt_1",
                "tilt_2",
                "phi_12",
                "phi_jl",
            ]

            waveform_params = self.GetWaveFormParamsFixed()

            for waveform_idx, suffix in enumerate(["_A", "_B"]):
                for parameter in spin_parameters:
                    key = f"{parameter}{suffix}"

                    if key in priors:
                        priors = self._fixPriorValue(
                            priors,
                            key,
                            waveform_params[waveform_idx][parameter],
                            True,
                        )

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
    def fixJointPriors(self, priors, useDeltaFunction):
        waveform_params = self.GetWaveFormParamsFixed()

        fixed_parameters = [
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
        ]

        for waveform_idx, suffix in enumerate(["_A", "_B"]):
            params = waveform_params[waveform_idx]

            for parameter in fixed_parameters:
                key = f"{parameter}{suffix}"

                if key not in priors:
                    continue

                value = params[parameter]

                priors = self._fixPriorValue(
                    priors,
                    key,
                    value,
                    useDeltaFunction,
                )

        # delta_t_AB is a joint parameter rather than an A/B parameter.
        if "delta_t_AB" in priors:
            delta_t = (
                waveform_params[0]["geocent_time"]
                - waveform_params[1]["geocent_time"]
            )

            priors = self._fixPriorValue(
                priors,
                "delta_t_AB",
                delta_t,
                useDeltaFunction,
            )

        return priors
    
    def _fixPriorValue(
        self,
        priors,
        key,
        value,
        useDeltaFunction,
    ):
        if useDeltaFunction:
            self.logger.info(
                f"$$$ making prior delta {key}"
            )

            priors[key] = DeltaFunction(
                value,
                name=key,
            )

            return priors

        if not self.configs["prior_controls"]["restrict_prior"]:
            return priors

        parameter_prior = priors[key]

        if (
            value > parameter_prior.maximum
            or value < parameter_prior.minimum
        ):
            raise ValueError(
                f"value: {value} not supported in prior "
                f"[{parameter_prior.minimum}, "
                f"{parameter_prior.maximum}] for {key}"
            )

        upperDiff = parameter_prior.maximum - value
        lowerDiff = value - parameter_prior.minimum

        priorRange = (
            parameter_prior.maximum
            - parameter_prior.minimum
        )

        restriction_str = self.configs[
            "prior_controls"
        ]["restriction_str"]

        tolerance = 1e-3

        delta = (
            restriction_str
            * min(upperDiff, lowerDiff)
        )

        if upperDiff < tolerance:
            priorMin = value - priorRange * restriction_str
            priorMax = value + delta

        elif lowerDiff < tolerance:
            priorMin = value - delta
            priorMax = value + priorRange * restriction_str

        else:
            priorMin = value - delta
            priorMax = value + delta

        parameter_prior.minimum = priorMin
        parameter_prior.maximum = priorMax

        self.logger.info(
            f"$$$ tightening prior {key}, "
            f"with bounds: {priorMin} and {priorMax}"
        )

        return priors
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
            luminosity_distance = 7500.0, #2000
            theta_jn            = 1.0, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 0.9,
            geocent_time        = self.configs["dataset_settings"]["T"]*0.8,# 0.5,
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
            luminosity_distance = 12000.0, #2000
            theta_jn            = 1.5, #angle of angular momentum
            psi                 = 2.659,  #angle of polarization
            phase               = 1.2,
            geocent_time        = self.configs["dataset_settings"]["T"]*0.8 - self.configs["scenario"]["time_delta"],
            ra                  = 1.2,
            dec                 = -1.2, 
        )
        injct_params_waves = [injct_params_wave_1,injct_params_wave_2]
        return injct_params_waves
    
    def _build_joint_prior(self):
        """Build the unmodified joint prior directly from the YAML config."""
        return bilby.core.prior.PriorDict(
            dictionary=self.prior_config
        )
        
    def _massesToChirpAndQ(self,m1, m2):
        # Ensure m1 >= m2 so that q = m2/m1 <= 1, as in bilby
        if m1 < m2:
            m1, m2 = m2, m1
        q = m2 / m1                       # mass_ratio in (0, 1]
        # chirp mass in solar masses
        chirp = (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)
        return chirp, q