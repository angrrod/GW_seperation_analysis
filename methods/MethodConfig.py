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
class MethodConfig:
    sampler:str           = "dynesty"
    nlive: int            = 4000 #4000
    dlogz: float          = 0.01 #0.1    #stopping criterion for the evidence
    sample: str           = "rslice" #rslice #unif', 'rwalk', 'slice', 'rslice', and 'auto' # performed until the autocorrelation length of the chain can be accurately determined.
    bound: str            = "multi"
    walks: int            = None #50          #steps for MCMC sampeler to select new candidates     
    nact: int             = 300  #300      #amount of steps is tuned so autocorr is small enough, needed for determining the correct slicing behaviour
    npool: int            = field(default_factory=_default_npool)   #18
    maxmcmc: int          = None #20000   #needed for MCMC sampeling, not needed for dynesty sampeling
    restrict_prior: bool  = True  # restrict the priors to a small value
    restriction_str:float = 0.1     #1  tunes the strenght of the restriction 
    use_deltas:bool       = False