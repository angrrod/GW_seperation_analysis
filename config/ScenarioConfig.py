from dataclasses import dataclass
@dataclass(frozen=True)
class ScenarioConfig:
    waveform_approximant: str  = "TaylorF2" #TaylorF2 "IMRPhenomXPHM" #IMRPhenomPv2  IMRPhenomD  IMRPhenomXPHM 
    minimum_frequency: float   = 10.0 #10
    sampling_frequency: float  = 2048.0 #4096
    reference_frequency: float = 20.0 #20
    duration: float            = 64.0 #64
    start_time:float           = 0.0
    time_delta: float          = 0.2 #time between the 2 signals
    ASD_file_name_ET: str      = "ET_D"
    ASD_file_name_CE: str      = "cosmic_explorer_strain"
    UseRelBinning: bool        = True 
    ineject_random_sample:bool = False  # generate a random or fixed signal-> needed for randomly testing
    generate_new_signal:bool   = True   # generate a new signal can be random or not,
                                        # needed to update the stored params
    scenarioId:str             = "test_correlation" #used for storing the PSD/noise strain
