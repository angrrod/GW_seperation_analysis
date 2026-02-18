from dataclasses import dataclass, field
import os

def _default_npool() -> int:
    """
    Pick a sensible worker pool size at runtime.
    Priority:
      1) GW_NPOOL (explicit override), needed to allow some CPU's for main process and not likelihood evals
      2) SLURM_CPUS_PER_TASK (match Slurm allocation)
      3) 18 default for local PC runs
      4) 1 (safe fallback)
    """
    v = os.environ.get("GW_NPOOL") or os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        n = int(v) if v is not None else 18  #here we run the normal one for HPC
    except ValueError:
        n = 1  #if error fall back to 1 to ensure runnability
    print(f"using {max(1, n)} processes")
    return max(1, n)

@dataclass(frozen=True)
class BaseSamplerConfig:
    sampler: str
    cores: int             = field(default_factory=_default_npool)
    # sampler-agnostic behaviour controls
    restrict_prior: bool   = True  #should be kept at false, doesn't work with TasNet/hierarchical pipeline as they need te have a different prior to allow for a support there
    restriction_str: float = 0.05
    use_deltas: bool       = False
    
@dataclass(frozen=True)
class DynestyConfig(BaseSamplerConfig):
    sampler:str           = "dynesty"
    nlive: int            = 1000 #1000
    dlogz: float          = 0.1 #0.1    #stopping criterion for the evidence
    sample: str           = "rslice" #rslice #unif', 'rwalk', 'slice', 'rslice', and 'auto' # performed until the autocorrelation length of the chain can be accurately determined.
    bound: str            = "multi"
    walks: int            = None #50          #steps for MCMC sampeler to select new candidates     
    nact: int             = None #300      #amount of steps is tuned so autocorr is small enough, needed for determining the correct slicing behaviour
    maxmcmc: int          = None #20000   #needed for MCMC sampeling, not needed for dynesty sampeling

@dataclass(frozen=True)
class PymcNutsConfig(BaseSamplerConfig):
    sampler:str          = "numpyro"
    sampler_name         = "NUTS"
    draws: int           = 10 #total steps = draws +tune
    tune: int            = 2000
    chains: int          = 1 #TODO:change
    target_accept: float = 0.8
    max_treedepth: int   = 10