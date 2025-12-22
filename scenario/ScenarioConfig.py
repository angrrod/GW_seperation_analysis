from dataclasses import dataclass
@dataclass(frozen=True)
class ScenarioConfig:
    waveform_approximant: str  = "IMRPhenomPv2"
    minimum_frequency: float   = 20.0 #10
    sampling_frequency: float  = 2048.0 #4096
    reference_frequency: float = 20.0 #10
    duration: float            = 32.0
    time_delta: float          = 3.0
    ASD_file_name_ET: str      = "ET_D"
    ASD_file_name_CE: str      = "cosmic_explorer_strain"
