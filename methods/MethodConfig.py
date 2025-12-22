from dataclasses import dataclass
@dataclass(frozen=True)
class MethodConfig:
    sampler:str  = "dynesty"
    nlive: int   = 100 #800
    dlogz: float = 5 #0.1         #stopping criterion for the evidence
    sample: str  = "rslice" #rslice #unif', 'rwalk', 'slice', 'rslice', and 'auto' # performed until the autocorrelation length of the chain can be accurately determined.
    bound: str   = "multi"
    walks: int   = None          #steps for MCMC sampeler to select new candidates     
    nact: int    = 50  #100      #amount of steps is tuned so autocorr is small enough, needed for determining the correct slicing behaviour
    npool: int   = 18
    maxmcmc: int = 20000   #needed for MCMC sampeling, not needed for dynesty sampeling 