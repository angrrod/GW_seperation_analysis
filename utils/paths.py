import os
from pathlib import Path

def get_base_log_dir() -> Path:
    return get_base_output_dir() / "logs"

def get_postprocessing_dir() -> Path:
    return get_base_output_dir() / "postProcessing"

def get_base_work_dir() -> Path:
    """
    Root directory for all generated artifacts (logs, postProcessing, results, …).
    """
    if "VSC_DATA" in os.environ:
        return Path(os.environ.get("GW_INP_DIR", os.environ["VSC_DATA"]))
    return Path(".")

def get_base_output_dir() -> Path:
    return get_base_work_dir() / "Output"

def get_dingo_dir() -> Path:
    return get_base_output_dir() / "Dingo"

def get_config_dir() -> Path:
    return get_base_work_dir() / "configs"

def get_scenario_data_dir() -> Path:
    return get_base_output_dir() / "scenarioData"

def get_dingo_dir_data() -> Path:
    if "VSC_SCRATCH" in os.environ:
        data_dir = Path(
            os.environ.get(
                "GW_DATA_DIR",
                Path(os.environ["VSC_SCRATCH"]) / "GW_separation" / "data",
            )
        )
    else:
        data_dir = get_dingo_dir() / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def get_cosmic_explorer_psd_path() -> Path:
    """
    Path to the Cosmic Explorer strain PSD file.

    Can be overridden explicitly with GW_PSD_FILE.
    """

    if "VSC_DATA" in os.environ:
        return (
            get_base_work_dir() / "cosmic_explorer_strain_PSD.txt"
        )

    return Path(
        "/root/phd/GW_seperation_analysis/"
        "cosmic_explorer_strain_PSD.txt"
    )
    
def get_waveform_dataset_path() -> Path:
    """
    Canonical path to the Dingo waveform dataset.

    On VSC this resolves through GW_DATA_DIR / VSC_SCRATCH.
    Locally it falls back to Output/Dingo/data.
    """
    return (
        get_dingo_dir_data()
        / "waveform_dataset.hdf5"
    )
    
def get_asd_dataset_path() -> Path:
    """
    Canonical path to the Dingo ASD dataset.

    On VSC this resolves through GW_DATA_DIR / VSC_SCRATCH.
    Locally it falls back to Output/Dingo/data.
    """
    return (
        get_dingo_dir_data()
        / "asd_dataset.hdf5"
    )

