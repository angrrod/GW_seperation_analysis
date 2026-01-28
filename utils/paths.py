import os
from pathlib import Path

def get_base_log_dir() -> Path:
    return get_base_work_dir() / "logs"

def get_postprocessing_dir() -> Path:
    return get_base_work_dir() / "postProcessing"

def get_base_work_dir() -> Path:
    """
    Root directory for all generated artifacts (logs, postProcessing, results, …).
    """
    if "VSC_DATA" in os.environ:
        return Path(os.environ.get("GW_INP_DIR", os.environ["VSC_DATA"]))
    return Path(".")