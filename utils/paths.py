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


def get_vsc_scratch_dir():
    """Resolve this project's scratch mount explicitly for the allocated cluster.

    VSC_SCRATCH may retain the generic /scratch/leuven alias. Use the known
    filesystem mount for wICE/Mindwell instead; retain the alias for other sites.
    """
    cluster = os.environ.get("SLURM_CLUSTER_NAME", os.environ.get("VSC_CLUSTER_NAME", "")).lower()
    roots = {
        "wice": Path("/lustre1/scratch/371/vsc37106"),
        "mindwell": Path("/gpfs1/scratch/371/vsc37106"),
    }
    if cluster in roots:
        return roots[cluster]
    scratch = os.environ.get("VSC_SCRATCH")
    return Path(scratch) if scratch else None


def get_dingo_dir_data() -> Path:
    """Use an explicit override, then the current cluster's native scratch.

    get_vsc_scratch_dir selects GPFS on Mindwell and Lustre on wICE. Datasets must be
    transferred when switching filesystems; do not silently select old copies
    from the other cluster. VSC_SCRATCH_LUSTRE1 / VSC_SCRATCH_GPFS1 locate those
    filesystems for transfers.
    """
    if os.environ.get("GW_DATA_DIR"):
        data_dir = Path(os.environ["GW_DATA_DIR"])
    elif get_vsc_scratch_dir() is not None:
        data_dir = get_vsc_scratch_dir() / "GW_separation" / "data"
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

    GW_WAVEFORM_DATASET_PATH overrides the dataset file explicitly.
    On VSC this otherwise resolves through GW_DATA_DIR / VSC_SCRATCH.
    Locally it falls back to Output/Dingo/data.
    """
    explicit_path = os.environ.get("GW_WAVEFORM_DATASET_PATH")
    if explicit_path:
        return Path(explicit_path)
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


def get_embedding_svd_cache_dir() -> Path:
    """Shared embedding preparation cache on the current cluster's scratch."""
    return Path(os.environ.get(
        "GW_EMBEDDING_SVD_CACHE_DIR", get_dingo_dir_data() / "embedding_svd"))

