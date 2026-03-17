import torch
from torch.distributions import MultivariateNormal, Categorical, MixtureSameFamily
import zuko
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from pyro.distributions.torch_distribution import TorchDistribution
from torch.distributions import constraints
import math
import numpy as np
from zuko.lazy import Flow, UnconditionalDistribution, UnconditionalTransform
from zuko.distributions import DiagNormal
from zuko.transforms import IdentityTransform

def plotMarg(Z,filename,d = 4):
    X_np = Z.detach().cpu().numpy()
    plt.figure()
    plt.scatter(X_np[0,:], X_np[1,:], s=d)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def sampleGaussianBase(params,N):
    paramsDerived   = buildDerivedParams(params)
    dist = torch.distributions.MultivariateNormal(loc=torch.tensor([0.,0.,0.,0.])
                                                , covariance_matrix=paramsDerived["Omega"])
    sample = dist.sample((N,))
    return sample

def sampleVectorCopula(params,N):
    eps  = 1e-11
    res  = sampleGaussianBase(params,N)  # replace with multivariate distr
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    Z    = stdn64.cdf(res.double()).clamp(eps, 1.0 - eps)
    return Z[:,0:2],Z[:,2:4]

def sampleVectorCopulaModel(params,N):
    Z_1,Z_2 = sampleVectorCopula(params,N)
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    T_1,T_2,_,_  = exctractFlows(params)
    if params["useIdentity"]:
        sample_1 = Z_1
        sample_2 = Z_2
    else:
        sample_1 = T_1(stdn64.icdf(Z_1).to(torch.float32)) #can be used because they are independent
        sample_2 = T_2(stdn64.icdf(Z_2).to(torch.float32))
    return torch.cat([sample_1,sample_2], dim=1).T

def logProbCopula(params,theta,d = 4):
    dtype  = torch.float64
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=dtype),
        torch.tensor(1., dtype=dtype),
    )
    paramsDerived   = buildDerivedParams(params)
    Sigma           = paramsDerived["Omega"].to(dtype=dtype)
    _, logabsdet    = torch.linalg.slogdet(Sigma)
    SigmaInv        = torch.linalg.inv(Sigma)
    PhiInv          = stdn64.icdf(theta)
    if len(theta.shape) == 2:
        einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (SigmaInv - torch.eye(d).to(dtype=dtype)), PhiInv)  #sum in order to deal with quadratic form dimensions
    else: #len = 1 so one event
        einsum = PhiInv.T @ (SigmaInv - torch.eye(d).to(dtype=dtype))@ PhiInv
    logDetTerm      = (-1/2)*logabsdet
    logCopulaTerm   = (-1/2)*einsum
    logDensity      = logDetTerm + logCopulaTerm
    return logDensity,logDetTerm,logCopulaTerm

def exctractFlows(params):
    dist_1 = params['flow_1']()
    T_1    = dist_1.transform
    dist_2 = params['flow_2']()
    T_2    = dist_2.transform
    return T_1,T_2,dist_1,dist_2

def logProbVectorCopula(params,theta):
    #allow for batch processing or single event:
    if len(theta.shape) == 2:
        theta02 = theta[:,:2]
        theta24 = theta[:,2:]
    else:
        theta02 = theta[:2]
        theta24 = theta[2:]
        
    eps = 1e-11 #numerical stability 
    stdn64 = torch.distributions.Normal( #-> cast inverse to get better precision
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    T_1,T_2,dist_1,dist_2  = exctractFlows(params)
    
    #get log prob of the flows using made products
    if params["useIdentity"]:
        logp_marg_1 = 0
        logp_marg_2 = 0
    else:
        logp_marg_1 = dist_1.log_prob(theta02)
        logp_marg_2 = dist_2.log_prob(theta24)
    
    #generate input for log prob copula
    if params["useIdentity"]:
        U_1 = theta02.clamp(eps, 1.0 - eps)
        U_2 = theta24.clamp(eps, 1.0 - eps)
    else:
        Q_1         = T_1.inv(theta02)
        Q_2         = T_2.inv(theta24)
        U_1         = stdn64.cdf(Q_1.double()).clamp(eps, 1.0 - eps)  #for numerical error
        U_2         = stdn64.cdf(Q_2.double()).clamp(eps, 1.0 - eps)
        
    if len(theta.shape) == 2:
        dim = 1
    else: #len = 1
        dim = 0
    U = torch.concat([U_1,U_2],dim = dim)
    logDensity,logDetTerm,logCopulaTerm  = logProbCopula(params,U)
    total = logp_marg_1 + logp_marg_2 + logDensity
    return total,logp_marg_1,logp_marg_2,logDensity,logDetTerm,logCopulaTerm

def getTrueParams():
    
    ### custom 4d distribution, mixture of 3 guassians with different correlation structure -> multimodal and one can tune the connection
    weights = torch.tensor([0.4, 0.6])

    # Means (2 components × 4 dimensions)
    means = torch.tensor([
        [0., 0.],
        [2., 2.],
    ])

    Sigma1 = torch.tensor([
        [1.0, 0.8],
        [0.8, 1.0]
    ])

    Sigma2 = torch.tensor([
        [1.0, 0.2],
        [0.2, 1.0],
    ])

    # Covariances (3 × 4 × 4)
    covs = torch.stack([
        Sigma1,
        Sigma2
    ])

    params_true_data = {
            "covs" : covs,
            "means" : means,
            "weights" : weights
            }
    # components = MultivariateNormal(loc=params_true_data['means'], covariance_matrix=params_true_data['covs'])
    # mixture = MixtureSameFamily(mix, params_true_data['weights'])
    # samp_mix = mixture.sample((10000,))
    # plotMarg(samp_mix.T)
    return params_true_data

def get_params(useIdentity:bool = False):
    params = {}
    # we start with a 4 d problem, get 2, 2d gaussians,
    # Use normalizing flows from U[0,1] distr to the gaussian, these are the quantiles
    d_1 = 2
    d_2 = 2
    flow_1 = zuko.flows.autoregressive.MAF(
        features=d_1,
        transforms=8,        # start ~4-8; 6 is a good default in 2D
        hidden_features=(128, 128),  # start 32-128
        randperm=True,       # important to mix dimensions between transforms
    )
    flow_2 = zuko.flows.autoregressive.MAF(
        features=d_2,
        transforms=8,
        hidden_features=(128, 128),
        randperm=True,
    )
    params['flow_1'] = flow_1
    params['flow_2'] = flow_2
    Sigma_21 = torch.tensor([
            [0.2, 0.4],
            [0.8, 0.1]
        ])
    params["Sigma_21"]      = pyro.param("Sigma_21", Sigma_21)
    params["Sigma_21_init"] = Sigma_21 
    params["useIdentity"]   = useIdentity
    return params

def ScaleSigma(Sigma_21):
    eps = 1e-12
    smax = torch.linalg.svdvals(Sigma_21).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    return Sigma_21 * (1.0 - eps) / (scale + eps)

def buildDerivedParams(params_orig):
    # build derived matrix
    params          = {}  
    #no constraint
    Sigma_21        = ScaleSigma(params_orig['Sigma_21'])
    top             = torch.cat([torch.eye(2), Sigma_21.T], dim=1)
    bottom          = torch.cat([Sigma_21, torch.eye(2)], dim=1)
    Omega           = torch.cat([top, bottom], dim=0)
    
    # Omega_bar       = params_orig['zeta']*torch.eye(4)+params_orig['B']@params_orig['B'].T
    # A               = torch.linalg.cholesky(Omega_bar)
    # Omega_new       = A @ Omega_bar @ A.T
    params["Omega"] = Omega
    params["SigmaScaled"] = Sigma_21
    #matrix will be pos definite if shur compliment I - A^T*A is positive definite which translates to |v|^2 > |Av|^2 for all v and is the case if the largest eigenvalue is smaller than 1 or the spectral norm is smaller than 1
    return params

def sample_true_data(N:int,true_params):
    z = Categorical(probs=true_params['weights']).sample((N,))          # (N,)

    # gather component parameters per datapoint
    mu    = true_params['means'][z]                                       # (N, d)
    Sigma = true_params['covs'][z]                                     # (N, d, d)

    # batch MVN: one MVN per datapoint
    X = MultivariateNormal(loc=mu, covariance_matrix=Sigma).sample()  # (N, d)
    return X

def make_identity_flow_module_2d():
    """
    Returns a Zuko lazy flow module.
    Calling this module with () returns a 2D NormalizingFlow distribution
    with identity transform and standard Gaussian base.
    """
    return Flow(
        transform=UnconditionalTransform(IdentityTransform),
        base=UnconditionalDistribution(
            DiagNormal,
            torch.zeros(2),
            torch.ones(2),
            buffer=True,
        ),
    )

def _fit_isotropic_kde_bandwidth_2d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: [N, 2]
    Scott-rule isotropic bandwidth for 2D KDE.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have shape [N,2], got {tuple(x.shape)}")

    n, d = x.shape
    std_per_dim = x.std(dim=0, unbiased=True)
    sigma = std_per_dim.mean().clamp_min(torch.tensor(eps, dtype=x.dtype, device=x.device))
    h = sigma * (n ** (-1.0 / (d + 4)))
    return h.clamp_min(torch.tensor(eps, dtype=x.dtype, device=x.device))

def _kde_log_prob_grid_2d(
    x_train: torch.Tensor,
    grid_points: torch.Tensor,
    h: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Evaluate isotropic Gaussian KDE log-density on a 2D grid.

    x_train      : [N, 2]
    grid_points  : [G, 2]
    returns      : [G]
    """
    n, d = x_train.shape
    if d != 2:
        raise ValueError(f"x_train must be [N,2], got {tuple(x_train.shape)}")
    if grid_points.ndim != 2 or grid_points.shape[1] != 2:
        raise ValueError(f"grid_points must be [G,2], got {tuple(grid_points.shape)}")

    log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * torch.log(h)
    out = torch.empty(grid_points.shape[0], dtype=x_train.dtype, device=x_train.device)

    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        g = grid_points[start:end]                  # [B, 2]
        diff = g[:, None, :] - x_train[None, :, :] # [B, N, 2]
        sqdist = (diff * diff).sum(dim=-1)         # [B, N]
        log_kernel = log_norm - 0.5 * sqdist / (h * h)
        out[start:end] = torch.logsumexp(log_kernel, dim=1) - math.log(n)

    return out

def plot_kde_3d_overlay_from_result(
    result: dict,
    dims=(0, 1),
    gridsize: int = 80,
    pad: float = 0.15,
    dtype: torch.dtype = torch.float64,
    chunk_size: int = 4096,
    filename: str | None = None,
    show_eval_points: bool = True,
    max_eval_points: int = 1000,
    elev=30,
    azim=45
):
    """
    Plot a 3D overlay of two 2D KDE marginal densities:
      - KDE fit on train_samples_2d[:, dims]
      - KDE fit on eval_samples_2d[:, dims]

    Parameters
    ----------
    result : dict
        Output from testInternalConsistency(...)
    dims : tuple[int, int]
        Which 2 of the 4 marginals to visualize.
    gridsize : int
        Number of grid points per axis.
    pad : float
        Fractional padding around data range.
    filename : str | None
        If given, save figure.
    show_eval_points : bool
        If True, scatter evaluation samples at their train-KDE height.
    """
    if "train_samples_2d" not in result or "eval_samples_2d" not in result:
        raise KeyError("result must contain 'train_samples_2d' and 'eval_samples_2d'.")

    i, j = dims
    x_train = result["train_samples_2d"][:, [i, j]].to(dtype=dtype)
    x_eval = result["eval_samples_2d"][:, [i, j]].to(dtype=dtype)

    if x_train.shape[1] != 2 or x_eval.shape[1] != 2:
        raise RuntimeError("Projected data is not 2D.")

    device = x_train.device

    # ------------------------------------------------------------------
    # Build plotting grid covering both sample clouds
    # ------------------------------------------------------------------
    all_x = torch.cat([x_train, x_eval], dim=0)
    mins = all_x.min(dim=0).values
    maxs = all_x.max(dim=0).values
    span = (maxs - mins).clamp_min(torch.tensor(1e-8, dtype=dtype, device=device))

    mins = mins - pad * span
    maxs = maxs + pad * span

    xs = torch.linspace(mins[0], maxs[0], gridsize, dtype=dtype, device=device)
    ys = torch.linspace(mins[1], maxs[1], gridsize, dtype=dtype, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    grid_points = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    # ------------------------------------------------------------------
    # Fit KDEs separately on projected train and eval sets
    # ------------------------------------------------------------------
    h_train = _fit_isotropic_kde_bandwidth_2d(x_train)
    h_eval = _fit_isotropic_kde_bandwidth_2d(x_eval)

    logZ_train = _kde_log_prob_grid_2d(
        x_train=x_train,
        grid_points=grid_points,
        h=h_train,
        chunk_size=chunk_size,
    )
    logZ_eval = _kde_log_prob_grid_2d(
        x_train=x_eval,
        grid_points=grid_points,
        h=h_eval,
        chunk_size=chunk_size,
    )

    Z_train = torch.exp(logZ_train).reshape(gridsize, gridsize).detach().cpu().numpy()
    Z_eval = torch.exp(logZ_eval).reshape(gridsize, gridsize).detach().cpu().numpy()

    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf1 = ax.plot_surface(
        X_np, Y_np, Z_train,
        alpha=0.60,
        linewidth=0,
        antialiased=True,
        label="KDE(train)"
    )
    ax.view_init(elev=elev, azim=azim)
    
    surf2 = ax.plot_surface(
        X_np, Y_np, Z_eval,
        alpha=0.45,
        linewidth=0,
        antialiased=True,
        label="KDE(eval)"
    )
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel(f"marginal {i}")
    ax.set_ylabel(f"marginal {j}")
    ax.set_zlabel("density")
    ax.set_title(f"2D marginal KDE overlay in dimensions ({i}, {j})")

    # matplotlib 3D surfaces do not show labels nicely in legends,
    # so create proxy artists
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(alpha=0.60, label="KDE(train)"),
        Patch(alpha=0.45, label="KDE(eval)"),
    ]
    if show_eval_points:
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], marker='o', linestyle='None', markersize=6, alpha=0.35, label="eval samples")
        )
    ax.legend(handles=legend_handles, loc="upper right")

    plt.tight_layout()
    plt.show()

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        
    return {
        "dims": dims,
        "bandwidth_train": float(h_train.cpu()),
        "bandwidth_eval": float(h_eval.cpu()),
    }

class VectorCopulaFlowQ(TorchDistribution):
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = False  # set True if you implement rsample()
    def __init__(self,params):
        self.params = params
        batch_shape = torch.Size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([4]), validate_args=None)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        total,_,_,_,_,_ = logProbVectorCopula(self.params,value)
        return total
    
    def sample(self, sample_shape=torch.Size()):
        if len(sample_shape) == 0:
            return sampleVectorCopulaModel(self.params, 1).T.squeeze(0) #return (4,)  # (4,)
        N = math.prod(sample_shape)
        return sampleVectorCopulaModel(self.params, N).T   #return (n,4)   
        
    def testInternalConsistency(
        self,
        N: int,
        M: int,
        chunk_size: int = 512,
        dtype: torch.dtype = torch.float64,
        verbose: bool = True,
        ):
        """
        Internal consistency test between `sample()` and `log_prob()`.

        Procedure
        ---------
        1. Draw N samples from the distribution and fit a Gaussian KDE to them.
        2. Draw a fresh independent set of M samples from the same distribution.
        3. Evaluate:
              - model_log_prob = self.log_prob(eval_samples)
              - kde_log_prob   = KDE log-density at eval_samples
        4. Compare both numerically.

        This is a smoke test, not a proof. KDE becomes weak in higher dimensions.

        Parameters
        ----------
        N : int
            Number of samples used to fit the KDE.
        M : int
            Number of fresh samples used for evaluation.
        chunk_size : int
            Chunk size for pairwise computations.
        dtype : torch.dtype
            Working dtype for KDE computations. float64 is recommended.
        verbose : bool
            If True, print summary statistics.

        Returns
        -------
        result : dict
            Dictionary containing train/eval samples, model log-probs, KDE log-probs,
            differences, and summary statistics.
        """
        if N < 2:
            raise ValueError("N must be at least 2 to fit a KDE.")
        if M < 1:
            raise ValueError("M must be at least 1.")

        # ------------------------------------------------------------
        # 1) Draw KDE fitting samples
        # ------------------------------------------------------------
        train_samples = self.sample((N,))
        if train_samples.ndim == 1:
            x_train = train_samples.unsqueeze(-1)
        else:
            x_train = train_samples.reshape(N, -1)
        x_train = x_train.to(dtype=dtype)

        device = x_train.device
        n, d = x_train.shape

        # ------------------------------------------------------------
        # 2) Draw independent evaluation samples
        # ------------------------------------------------------------
        eval_samples = self.sample((M,))
        if eval_samples.ndim == 1:
            x_eval = eval_samples.unsqueeze(-1)
        else:
            x_eval = eval_samples.reshape(M, -1)
        x_eval = x_eval.to(dtype=dtype)

        m = x_eval.shape[0]
        if x_eval.shape[1] != d:
            raise RuntimeError(
                f"Dimension mismatch: train samples have dim {d}, "
                f"eval samples have dim {x_eval.shape[1]}."
            )
        # plotMarg(x_eval.T[1:3,:],filename = "toy_problem/samplesCopula[0:2,:]")
        # plotMarg(x_eval.T[(0,3),:],filename = "toy_problem/samplesCopula[2:4,:]")
        
        # ------------------------------------------------------------
        # 3) Model log_prob on fresh evaluation points
        # ------------------------------------------------------------
        model_log_prob = self.log_prob(eval_samples).detach().to(dtype=dtype)
        if model_log_prob.ndim != 1 or model_log_prob.shape[0] != M:
            raise RuntimeError(
                f"Expected self.log_prob(eval_samples) to return shape ({M},), "
                f"but got {tuple(model_log_prob.shape)}."
            )

        # ------------------------------------------------------------
        # 4) Fit isotropic Gaussian KDE using Scott's rule
        #    h = sigma * n^{-1/(d+4)}
        # ------------------------------------------------------------
        std_per_dim = x_train.std(dim=0, unbiased=True)
        sigma = std_per_dim.mean().clamp_min(
            torch.tensor(1e-8, dtype=dtype, device=device)
        )
        h = sigma * (n ** (-1.0 / (d + 4)))
        h = h.clamp_min(torch.tensor(1e-8, dtype=dtype, device=device))

        # log Gaussian kernel normalization:
        # log N(x | mu, h^2 I) = -d/2 log(2pi) - d log h - ||x-mu||^2/(2h^2)
        log_norm = -0.5 * d * math.log(2.0 * math.pi) - d * torch.log(h)

        # ------------------------------------------------------------
        # 5) Evaluate KDE on fresh evaluation points
        #
        # log \hat p(x)
        # = log(1/N * sum_j N(x | x_train_j, h^2 I))
        # ------------------------------------------------------------
        kde_log_prob = torch.empty(m, dtype=dtype, device=device)

        for start in range(0, m, chunk_size):
            end = min(start + chunk_size, m)
            x_chunk = x_eval[start:end]  # [B, D]

            diff = x_chunk[:, None, :] - x_train[None, :, :]  # [B, N, D]
            sqdist = (diff * diff).sum(dim=-1)                # [B, N]

            log_kernel = log_norm - 0.5 * sqdist / (h * h)    # [B, N]
            kde_log_prob[start:end] = torch.logsumexp(log_kernel, dim=1) - math.log(n)

        # ------------------------------------------------------------
        # 6) Compare
        # ------------------------------------------------------------
        difference = model_log_prob - kde_log_prob
        abs_difference = difference.abs()
        sq_difference = difference ** 2

        if m > 1:
            corrcoef = float(
                torch.corrcoef(torch.stack([model_log_prob, kde_log_prob]))[0, 1].cpu()
            )
        else:
            corrcoef = float("nan")

        result = {
            "train_samples": train_samples,
            "eval_samples": eval_samples,
            "train_samples_2d": x_train,
            "eval_samples_2d": x_eval,
            "model_log_prob": model_log_prob,
            "kde_log_prob": kde_log_prob,
            "difference": difference,
            "abs_difference": abs_difference,
            "summary": {
                "N_train": n,
                "M_eval": m,
                "dim": d,
                "bandwidth": float(h.cpu()),
                "model_log_prob_mean": float(model_log_prob.mean().cpu()),
                "kde_log_prob_mean": float(kde_log_prob.mean().cpu()),
                "difference_mean": float(difference.mean().cpu()),
                "difference_std": float(difference.std(unbiased=True).cpu()) if m > 1 else 0.0,
                "difference_median": float(difference.median().cpu()),
                "abs_difference_mean": float(abs_difference.mean().cpu()),
                "abs_difference_median": float(abs_difference.median().cpu()),
                "abs_difference_max": float(abs_difference.max().cpu()),
                "rmse": float(torch.sqrt(sq_difference.mean()).cpu()),
                "corrcoef": corrcoef,
            },
        }

        if verbose:
            s = result["summary"]
            print("=== Internal consistency test ===")
            print(f"N train                : {s['N_train']}")
            print(f"M eval                 : {s['M_eval']}")
            print(f"dim                    : {s['dim']}")
            print(f"KDE bandwidth          : {s['bandwidth']:.6g}")
            print(f"mean log_prob(model)   : {s['model_log_prob_mean']:.6f}")
            print(f"mean log_prob(KDE)     : {s['kde_log_prob_mean']:.6f}")
            print(f"mean difference        : {s['difference_mean']:.6f}")
            print(f"std difference         : {s['difference_std']:.6f}")
            print(f"median difference      : {s['difference_median']:.6f}")
            print(f"mean |difference|      : {s['abs_difference_mean']:.6f}")
            print(f"median |difference|    : {s['abs_difference_median']:.6f}")
            print(f"max |difference|       : {s['abs_difference_max']:.6f}")
            print(f"RMSE                   : {s['rmse']:.6f}")
            print(f"corr(model, KDE)       : {s['corrcoef']:.6f}")
        
        plt.figure(figsize=(6,6))
        plt.scatter(kde_log_prob.detach().numpy(), model_log_prob.detach().numpy(), s=4, alpha=0.3)
        mn = min(kde_log_prob.detach().numpy().min(), model_log_prob.detach().numpy().min()).item()
        mx = max(kde_log_prob.detach().numpy().max(), model_log_prob.detach().numpy().max()).item()
        plt.plot([mn, mx], [mn, mx], 'r--')
        plt.xlabel("KDE log prob")
        plt.ylabel("Model log prob")
        plt.title("Model vs KDE log-density")
        plt.show()
        return result
    
#perform SVI optimization
class pyroCopulaSVIPipeline:
    def __init__(self,batch_size = 512):
        self.trueParams  = getTrueParams()
        self.modelParams = get_params()
        self.batch_size  = batch_size
        
    def model(self,data):
                
        #priors
        theta = pyro.sample(
            "theta",
            dist.MultivariateNormal(
                loc=torch.zeros(4),
                covariance_matrix=5.0 * torch.eye(4)
            )
        )
        mu1 = theta[:2]
        mu2 = theta[2:]
        
        mix = dist.Categorical(self.trueParams["weights"])
        comp = dist.MultivariateNormal(
            loc=torch.stack([mu1, mu2], dim=0),  # shape (2, 2)
            covariance_matrix=self.trueParams["covs"]     # shape (2, 2, 2)
        ) 
        gmm = dist.MixtureSameFamily(mix, comp)
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", gmm, obs=data) #gmm(theta)

    def guide(self,data):
        pyro.module("flow_1", self.modelParams['flow_1'])
        pyro.module("flow_2", self.modelParams['flow_2'])
        
        Sigma_21 = pyro.param("Sigma_21", self.modelParams["Sigma_21_init"])
        
        params_now = dict(self.modelParams)
        params_now["Sigma_21"] = Sigma_21  #force the update
        
        #sample from 4d joint distr, and cast them to the 2 variables
        q     = VectorCopulaFlowQ(params_now)
        pyro.sample("theta", q)
        
def get_params_now(pipe):
    # pipe.modelParams contains references to flow modules etc.
    params_now = dict(pipe.modelParams)

    # overwrite anything learned via pyro.param
    params_now["Sigma_21"] = pyro.param("Sigma_21")

    # (optional) if you later move to other pyro.params, add them here too
    return params_now
    
def runModel(nSteps,printEvery,N,svi,X,pipe):
    step       = 0
    prev_loss  = 0
    diff       = 0
    loss_list = []
    diagnostics = []
    
    while (step < nSteps):
        print(f"step: {step}")
        loss = svi.step(X)        

        diff = prev_loss - loss
        loss_list.append(loss)
        params_now = get_params_now(pipe)
        with torch.no_grad():
            guide_trace = pyro.poutine.trace(pipe.guide).get_trace(X)
            theta_sample = guide_trace.nodes["theta"]["value"].detach()
            total, m1, m2, cop,logDetTerm,logCopulaTerm = logProbVectorCopula(params_now, theta_sample)
            diag = dict(
                total  = total.mean().item(),
                marg1  = m1.mean().item(),
                marg2  = m2.mean().item(),
                copula = cop.mean().item(),
                det    = logDetTerm.mean().item(),
                gauss  = logCopulaTerm.mean().item(),
            )
            diagnostics.append(diag)
            
        if step % printEvery == 0:                
            PyroParams = {"Sigma_21": pyro.param("Sigma_21")}
            derived    = buildDerivedParams(PyroParams)
            Omega      = derived["Omega"]
            eigvals    = torch.linalg.eigvalsh(Omega)
            print("")
            print(f"step: {step}, diff: {diff}")
            print(f"loss: {loss}, loss/N: {loss / N}")
            print(f"Omega: {Omega}")
            print(f"eigvals: {eigvals}")
        prev_loss = loss
        step +=1
    return loss_list,diagnostics
    
def postProcessTest(loss_list,diagnostics,pipe,X):
    # ---- plot diagnostic curve ----
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("SVI loss difference diagnostic")
    plt.grid(True)
    filename = "svi_loss_diagnostic.png"
    plt.savefig("toy_problem/"+filename, dpi=300, bbox_inches="tight")
    
    #logprob plots
    total  = np.array([d["total"]  for d in diagnostics])
    marg1  = np.array([d["marg1"]  for d in diagnostics])
    marg2  = np.array([d["marg2"]  for d in diagnostics])
    det    = np.array([d["det"] for d in diagnostics])
    gauss  = np.array([d["gauss"] for d in diagnostics])
    copula = np.array([d["copula"] for d in diagnostics])
    steps  = range(len(total))
    filename = "loss_diagnostics"
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Left axis → component log-densities
    ax1.plot(steps, marg1, label="marg1", linewidth=2)
    ax1.plot(steps, marg2, label="marg2", linewidth=2)
    ax1.plot(steps, copula, label="copula", linewidth=2)
    ax1.plot(steps, copula, label="copula", linewidth=2)
    ax1.plot(steps, det, label="det part of copula", linewidth=2)
    ax1.plot(steps, gauss, label="gauss part of copula", linewidth=2)
    ax1.plot(steps, total, label="total", linewidth=3, linestyle="--")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean log-density components")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    plt.title("SVI Diagnostics: Marginals, Copula, and Loss")
    plt.tight_layout()
    plt.savefig("toy_problem/"+filename, dpi=300, bbox_inches="tight")

    #test 1 log_prob
    params_now = dict(pipe.modelParams)
    params_now["Sigma_21"] = pyro.param("Sigma_21")

    q = VectorCopulaFlowQ(params_now)
    x = q.sample((4096,))
    lp = q.log_prob(x)

    print("finite log_prob on own samples:", torch.isfinite(lp).all().item())
    print("lp mean/std:", lp.mean().item(), lp.std().item())
    
    #test 2 -> correct T use
    params_now = get_params_now(pipe)
    q = VectorCopulaFlowQ(params_now)

    theta = q.sample((8192,))   # shape (8192, 4)
    mu1 = theta[:, 0:2]
    mu2 = theta[:, 2:4]

    T1 = params_now["flow_1"]().transform
    T2 = params_now["flow_2"]().transform

    z1 = T1.inv(mu1)
    z2 = T2.inv(mu2)

    print(f"mu1 shape: {mu1.shape}, mu2 shape: {mu2.shape}")
    print(f"T1^-1(mu1): mean {z1.mean(0)}, var {z1.var(0)}")
    print(f"T2^-1(mu2): mean {z2.mean(0)}, var {z2.var(0)}")
    
    # 3 marginals
    params_now = dict(pipe.modelParams)
    params_now["Sigma_21"] = pyro.param("Sigma_21")
    q = VectorCopulaFlowQ(params_now)

    samples = q.sample((5000,))   # (5000, 4)

    mu1_samples = samples[:, 0:2]
    mu2_samples = samples[:, 2:4]

    plotMarg(mu1_samples.T, filename="toy_problem/mu1_samples")
    plotMarg(mu2_samples.T, filename="toy_problem/mu2_samples")
    plotMarg(X.T, filename="toy_problem/data")
    
def main():
    pyro.clear_param_store()
    N          = 8192
    batch_size = 512
    nSteps     = 100
    printEvery = 50
    pipe       = pyroCopulaSVIPipeline(batch_size=batch_size)
    
    #standardize the data
    X          = sample_true_data(N, pipe.trueParams)
    mu         = X.mean(0)
    std        = X.std(0)
    X          = (X - mu) / std
    svi        = SVI(pipe.model, pipe.guide, ClippedAdam({"lr": 1e-5, "clip_norm": 5.0}), loss=Trace_ELBO())
    
    loss_list,diagnostics = runModel(nSteps,printEvery,N,svi,X,pipe)
    postProcessTest(loss_list,diagnostics,pipe,X)
    
if __name__ == "__main__":
    main()
    # ModelTest = VectorCopulaFlowQ(get_params(useIdentity = True))
    # result = ModelTest.testInternalConsistency(10000,10000)
    print("end")
    #TODO: fix model -> sample from ML model