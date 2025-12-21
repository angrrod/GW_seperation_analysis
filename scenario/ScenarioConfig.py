from dataclasses import dataclass
@dataclass(frozen=True)
class ScenarioConfig:
    waveform_approximant: str  = "IMRPhenomPv2"
    minimum_frequency: float   = 10.0 #10
    sampling_frequency: float  = 4096.0
    reference_frequency: float = 10.0 #10
    duration: float            = 64.0
    time_delta: float          = 3.0
    ASD_file_name: str         = "ET_D"