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
import os
from datetime import datetime
import torch
import torch.nn as nn
from torch.distributions import Transform, constraints
import torch.nn.functional as F
from torch.distributions.constraints import real
from torch.distributions import AffineTransform
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

class FullAffineTransform(Transform, nn.Module):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __init__(self, loc, raw_L, eps=1e-4):
        Transform.__init__(self)
        nn.Module.__init__(self)

        self.loc = nn.Parameter(loc.clone())
        self.raw_L = nn.Parameter(raw_L.clone())
        self.eps = eps

    def _matrix(self):
        L = torch.tril(self.raw_L)
        diag = torch.diagonal(L, 0)
        diag = F.softplus(diag) + self.eps
        L = L - torch.diag(torch.diagonal(L, 0)) + torch.diag(diag)
        return L

    def _call(self, x):
        L = self._matrix()
        if x.ndim == 1:
            return self.loc + L @ x
        return x @ L.T + self.loc

    def _inverse(self, y):
        L = self._matrix()

        if y.ndim == 1:
            rhs = (y - self.loc).unsqueeze(-1)   # (d,1)
            sol = torch.linalg.solve_triangular(L, rhs, upper=False)
            return sol.squeeze(-1)
        else:
            rhs = (y - self.loc).T               # (d,N)
            sol = torch.linalg.solve_triangular(L, rhs, upper=False)
            return sol.T

    def log_abs_det_jacobian(self, x, y):
        L = self._matrix()
        ladj = torch.log(torch.diagonal(L, 0)).sum()

        if x.ndim == 1:
            return ladj
        return ladj.expand(x.shape[0])

def make_affine_flow_module_2d(
    loc_init=None,
    log_scale_init=None,
    dim = 2
):
    """
    Returns a Zuko lazy flow module.
    Calling this module with () returns a 2D NormalizingFlow distribution
    with an elementwise affine transform and standard Gaussian base.

    Forward map:
        x = loc + scale * z
    where z ~ N(0, I), scale = exp(log_scale) > 0

    This is the simplest nontrivial normalizing flow:
    - invertible
    - trainable
    - analytically tractable
    """
    if loc_init is None:
        loc_init = torch.zeros(dim)
    if log_scale_init is None:
        log_scale_init = 0.01 * torch.randn(2, 2)

    return Flow(
        transform = UnconditionalTransform(
            FullAffineTransform,
            loc    = loc_init,
            raw_L  = log_scale_init,  #needs to be positive
            buffer = False,   # important: loc/scale become trainable parameters
        ),
        base=UnconditionalDistribution(
            DiagNormal,
            torch.zeros(dim),
            torch.ones(dim),
            buffer=True,
        ),
    )

#continuous clamp
def stable_unit_interval(a, eps=1e-6):
    return torch.clamp(a, eps, 1 - eps) #eps + (1 - 2 * eps) * torch.sigmoid(a)

def plotMarg(Z,filename,d = 4):
    X_np = Z.detach().cpu().numpy()
    # if len(X_np.shape) == 1:
        
    plt.figure()
    plt.scatter(X_np[0,:], X_np[1,:], s=d)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def sampleGaussianBase(params,N):
    paramsDerived   = buildDerivedParams(params)
    dist = torch.distributions.MultivariateNormal(loc=torch.tensor([0.,0.,0.,0.])
                                                , covariance_matrix=paramsDerived["Omega"])
    sample = dist.rsample((N,))
    return sample

def sampleVectorCopula(params,N,useGaussianBase: bool):
    res  = sampleGaussianBase(params,N)  # replace with multivariate distr
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    if useGaussianBase:
        Z    = stable_unit_interval(stdn64.cdf(res.double()))
    else:
        Z    = res
    return Z[:,0:2],Z[:,2:4]

def sampleVectorCopulaModel(params,N,useGaussianBase:bool = False):
    Z_1,Z_2 = sampleVectorCopula(params,N,useGaussianBase)
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    T_1,T_2,_,_  = exctractFlows(params)
    if params["useIdentity"]:
        if useGaussianBase:
            raise ValueError("when using the identity we cannot use a gaussian base")
        sample_1 = Z_1
        sample_2 = Z_2
    else:
        if useGaussianBase:
            sample_1 = T_1(stdn64.icdf(Z_1).to(torch.float32)) #can be used because they are independent
            sample_2 = T_2(stdn64.icdf(Z_2).to(torch.float32))
        else:
            sample_1 = T_1(Z_1)
            sample_2 = T_2(Z_2)
    return torch.cat([sample_1,sample_2], dim=1).T

def logProbCopula(params,Q,d = 4):
    dtype  = torch.float64
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=dtype),
        torch.tensor(1., dtype=dtype),
    )
    paramsDerived   = buildDerivedParams(params)
    Omega           = paramsDerived["Omega"].to(dtype=dtype)
    I               = torch.eye(4).to(dtype=dtype)
    _, logabsdet    = torch.linalg.slogdet(Omega)
    OmegaInv        = torch.linalg.solve(Omega, I) 
    PhiInv          = Q
    if len(Q.shape) == 2:
        einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (OmegaInv- I), PhiInv)  #sum in order to deal with quadratic form dimensions
    else: #len = 1 so one event
        einsum = PhiInv.T @ (OmegaInv- I) @ PhiInv
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

def logProbVectorCopula(params,theta,useGaussianBase:bool = False):
    #allow for batch processing or single event:
    if len(theta.shape) == 2:
        theta02 = theta[:,:2]
        theta24 = theta[:,2:]
    else:
        theta02 = theta[:2]
        theta24 = theta[2:]
        
    # stdn64 = torch.distributions.Normal( #-> cast inverse to get better precision
    #     torch.tensor(0., dtype=torch.float64),
    #     torch.tensor(1., dtype=torch.float64),
    # )
    T_1,T_2,dist_1,dist_2  = exctractFlows(params)
    
    #get log prob of the marginal flows using made products
    if params["useIdentity"]:
        logp_marg_1 = torch.tensor(0,dtype = torch.float64)
        logp_marg_2 = torch.tensor(0,dtype = torch.float64)
    else:
        logp_marg_1 = dist_1.log_prob(theta02)
        logp_marg_2 = dist_2.log_prob(theta24)
    
    #generate input for log prob copula
    if params["useIdentity"]:
        Q_1 = theta02.to(dtype=torch.float64)
        Q_2 = theta24.to(dtype=torch.float64)
    else:
        Q_1         = T_1.inv(theta02)
        Q_2         = T_2.inv(theta24)
        
    if len(theta.shape) == 2:
        dim = 1
    else: #len = 1
        dim = 0
    Q = torch.concat([Q_1,Q_2],dim = dim).to(dtype=torch.float64)
    
    logDensity,logDetTerm,logCopulaTerm  = logProbCopula(params,Q)
    total = logp_marg_1 + logp_marg_2 + logDensity
    return total,logp_marg_1,logp_marg_2,logDensity,logDetTerm,logCopulaTerm

def getTrueParams():
    
    ### custom 4d distribution, mixture of 3 guassians with different correlation structure -> multimodal and one can tune the connection
    #weights = torch.tensor([0.4, 0.6])

    # Means (2 components × 4 dimensions)
    # means = torch.tensor([
    #     [0., 0.],
    #     [2., 2.],
    # ])
    means = torch.tensor([0.0,0.0,0.0,0.0])
    # Sigma1 = torch.tensor([
    #     [1.0, 0.8],
    #     [0.8, 1.0]
    # ])

    # Sigma2 = torch.tensor([
    #     [1.0, 0.2],
    #     [0.2, 1.0],
    # ])

    # # Covariances (3 × 4 × 4)
    # covs = torch.stack([
    #     Sigma1,
    #     Sigma2
    # ])
    
    covs = torch.tensor([
        [1.0, 0.6, 0.2, 0.1],
        [0.6, 1.0, 0.05, 0.5],
        [0.2, 0.05, 1.0, 0.45],
        [0.1, 0.5, 0.45, 1.0],
    ])

    params_true_data = {
            "covs" : covs,
            "means" : means,
            #"weights" : weights,
            "priorVar" : 1
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
    # flow_1 = zuko.flows.NSF(
    #     features=d_1,
    #     transforms=4,
    #     context=0,
    #     hidden_features=(8,),
    #     bins=8,
    #     randperm=True,
    # )

    # flow_2 = zuko.flows.NSF(
    #     features=d_2,
    #     transforms=4,
    #     context=0,
    #     hidden_features=(8,),
    #     bins=8,
    #     randperm=True,
    # )
    
    if useIdentity:
        flow_1 = make_identity_flow_module_2d()
        flow_2 = make_identity_flow_module_2d()
    else:
        flow_1 = make_affine_flow_module_2d(
            loc_init=torch.zeros(d_1),
            log_scale_init= torch.randn(2, 2)#torch.zeros(d_1,d_1),
        )
        flow_2 = make_affine_flow_module_2d(
            loc_init=torch.zeros(d_2),
            log_scale_init= torch.randn(2, 2)#torch.zeros(d_2,d_2),
        )
    params['flow_1'] = flow_1
    params['flow_2'] = flow_2
    B = torch.eye(4)#torch.tensor([
        #     [0.2, 0.4, 0.0, 1.0],
        #     [0.8, 0.1, 0.2, 0.3],
        #     [1.2, 0.4, 1.0, 0.0],
        #     [1.4, 0.0, 0.1, 1.0]
        # ])
    params["B"]             = B
    params["B_init"]        = B.clone().detach() 
    params['zeta']          = torch.tensor(1.0)
    params["useIdentity"]   = useIdentity
    return params

def ScaleSigma(Sigma,rho = 0.9999):
    eps = 1e-6
    smax = torch.linalg.svdvals(Sigma).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    return Sigma* rho * (1.0 - eps) / (scale + eps)

def buildDerivedParams(params_orig):
    # build derived matrix
    params          = {}  
    
    # no constraint
    # top             = torch.cat([torch.eye(2), Sigma_21.T], dim=1)
    # bottom          = torch.cat([Sigma_21, torch.eye(2)], dim=1)
    # Omega           = torch.cat([top, bottom], dim=0)
    B               = params_orig['B']
    OmegaBar        = params_orig['zeta']*torch.eye(4) + B @ B.T
    dimList         = [2,2]
    L               = torch.linalg.cholesky(Blockdiag(OmegaBar,dimList))
    I               = torch.eye(L.shape[-1], device=L.device, dtype=L.dtype)
    A               = torch.linalg.solve_triangular(L, I, upper=False)
    Omega           = A @ OmegaBar @ A.T
    # Omega           = ScaleSigma(Omega)
    params["Omega"] = Omega
    #matrix will be pos definite if shur compliment I - A^T*A is positive definite which translates to |v|^2 > |Av|^2 for all v and is the case if the largest eigenvalue is smaller than 1 or the spectral norm is smaller than 1
    return params

def Blockdiag(B,dimList):
    Bdiag = B.clone()
    prevDim = 0
    for dim in dimList:
        ind = dim + prevDim
        Bdiag[:ind,ind:] = 0
        Bdiag[ind:,:ind] = 0   
        prevDim = dim  
    return Bdiag

def sample_true_data(N:int,true_params):
    # z = Categorical(probs=true_params['weights']).sample((N,))          # (N,)
    
    # gather component parameters per datapoint
    mu    = true_params['means']#[z]                                       # (N, d)
    Sigma = true_params['covs']#[z]                                     # (N, d, d)

    # batch MVN: one MVN per datapoint
    X = MultivariateNormal(loc=mu, covariance_matrix=Sigma).sample((N,))  # (N, d)
    return X

def plot_posterior_marginals(save_dir="toy_problem", filename="true_marginals.png"):

    os.makedirs(save_dir, exist_ok=True)
    mu = torch.zeros(4)
    cov = getPosteriorSigma(2048)
    dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    X = dist.sample((5000,))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # First marginal: dims 0 and 1
    axes[0].scatter(X[:, 0], X[:, 1], s=5, alpha=0.5)
    axes[0].set_title("Marginal [0,1]")
    axes[0].set_xlabel("x0")
    axes[0].set_ylabel("x1")

    # Second marginal: dims 2 and 3
    axes[1].scatter(X[:, 2], X[:, 3], s=5, alpha=0.5)
    axes[1].set_title("Marginal [2,3]")
    axes[1].set_xlabel("x2")
    axes[1].set_ylabel("x3")

    plt.tight_layout()
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {path}")
    
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

def kde_log_prob_scipy(x_train: torch.Tensor, x_eval: torch.Tensor, bw_method="scott", eps=1e-300):
    x_train_np = x_train.detach().cpu().numpy().T   # (d, N)
    x_eval_np = x_eval.detach().cpu().numpy().T     # (d, M)

    kde = gaussian_kde(x_train_np, bw_method=bw_method)
    dens = kde(x_eval_np)
    logdens = np.log(np.clip(dens, eps, None))

    return (
        torch.from_numpy(logdens).to(dtype=x_train.dtype, device=x_train.device),
        kde
    )
    
def plot_kde_3d_overlay_from_result(
    result: dict,
    dims=(0, 1),
    gridsize: int = 80,
    pad: float = 0.15,
    dtype: torch.dtype = torch.float64,
    filename: str | None = None,
    show_eval_points: bool = True,
    max_eval_points: int = 1000,
    elev=10,
    azim=10,
    bw_method="scott",
):
    """
    Plot a 3D overlay of two 2D KDE marginal densities using scipy.stats.gaussian_kde:
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
        If True, scatter a subset of eval samples on z=0.
    max_eval_points : int
        Maximum number of eval points to show.
    bw_method : str | scalar | callable
        Bandwidth method passed to scipy.stats.gaussian_kde.
        Typical choices: "scott", "silverman", or a scalar multiplier.
    """
    if "train_samples_2d" not in result or "eval_samples_2d" not in result:
        raise KeyError("result must contain 'train_samples_2d' and 'eval_samples_2d'.")

    i, j = dims
    x_train = result["train_samples_2d"][:, [i, j]].to(dtype=dtype)
    x_eval = result["eval_samples_2d"][:, [i, j]].to(dtype=dtype)

    if x_train.ndim != 2 or x_train.shape[1] != 2:
        raise RuntimeError(f"Projected train data must be shape [N,2], got {tuple(x_train.shape)}")
    if x_eval.ndim != 2 or x_eval.shape[1] != 2:
        raise RuntimeError(f"Projected eval data must be shape [M,2], got {tuple(x_eval.shape)}")

    # Convert to numpy for scipy
    x_train_np = x_train.detach().cpu().numpy()   # shape (N, 2)
    x_eval_np = x_eval.detach().cpu().numpy()     # shape (M, 2)

    # ------------------------------------------------------------------
    # Build plotting grid covering both sample clouds
    # ------------------------------------------------------------------
    all_x = np.vstack([x_train_np, x_eval_np])
    mins = all_x.min(axis=0)
    maxs = all_x.max(axis=0)
    span = np.maximum(maxs - mins, 1e-8)

    mins = mins - pad * span
    maxs = maxs + pad * span

    xs = np.linspace(mins[0], maxs[0], gridsize)
    ys = np.linspace(mins[1], maxs[1], gridsize)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    grid_points = np.vstack([X.ravel(), Y.ravel()])   # shape (2, G)

    # ------------------------------------------------------------------
    # Fit KDEs separately on projected train and eval sets
    # scipy gaussian_kde expects shape (d, N)
    # ------------------------------------------------------------------
    kde_train = gaussian_kde(x_train_np.T, bw_method=bw_method)  # sample
    kde_eval = gaussian_kde(x_eval_np.T, bw_method=bw_method)    # log prob

    Z_train = kde_train(grid_points).reshape(gridsize, gridsize)
    Z_eval = kde_eval(grid_points).reshape(gridsize, gridsize)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        X, Y, Z_train, #sample
        alpha=0.20,
        linewidth=0,
        antialiased=True,
        color = 'r'
    )
    ax.plot_surface(
        X, Y, Z_eval, #log prob
        alpha=0.55,
        linewidth=0,
        antialiased=True,
    )

    if show_eval_points and len(x_eval_np) > 0:
        if len(x_eval_np) > max_eval_points:
            idx = np.random.choice(len(x_eval_np), size=max_eval_points, replace=False)
            pts = x_eval_np[idx]
        else:
            pts = x_eval_np

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            np.zeros(len(pts)),
            s=4,
            alpha=0.35,
        )

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel(f"marginal {i}")
    ax.set_ylabel(f"marginal {j}")
    ax.set_zlabel("density")
    ax.set_title(f"2D marginal KDE overlay in dimensions ({i}, {j})")

    # legend proxies
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

    if filename is not None:
        plt.savefig(filename, dpi=200, bbox_inches="tight")

    return {
        "dims": dims,
        "bw_method": bw_method,
        "bandwidth_factor_train": float(kde_train.factor),
        "bandwidth_factor_eval": float(kde_eval.factor),
    }

class VectorCopulaFlowQ(TorchDistribution):
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = True  # set True if you implement rsample()
    def __init__(self,params):
        self.params = params
        batch_shape = torch.Size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([4]), validate_args=None)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        total,_,_,_,_,_ = logProbVectorCopula(self.params,value)
        return total
    
    def rsample(self, sample_shape=torch.Size()):
        if len(sample_shape) == 0:
            return sampleVectorCopulaModel(self.params, 1).T.squeeze(0) #return (4,)  # (4,)
        N = math.prod(sample_shape)
        return sampleVectorCopulaModel(self.params, N).T   #return (n,4)   
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
    
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
        kde_log_prob, kde = kde_log_prob_scipy(x_train, x_eval, bw_method="scott")
        
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
            "train_samples": train_samples,  #sample
            "eval_samples": eval_samples,    #log prob
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
                "bandwidth_factor": float(kde.factor),
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
        plt.savefig("toy_problem/model_vs_kde.png", dpi=300)
        plot_kde_3d_overlay_from_result(result,filename = 'toy_problem/kde_overaly_3d')
        return result
    
#perform SVI optimization
class pyroCopulaSVIPipeline:
    def __init__(self,batch_size = 512, useIdentity = False):
        self.trueParams  = getTrueParams()
        self.modelParams = get_params(useIdentity = useIdentity)
        self.batch_size  = batch_size
        
    def _regularize(self,lam = 1e-5):
        reg = 0.0
        for name, param in self.modelParams["flow_1"].named_parameters():
            reg = reg + (param ** 2).sum()

        for nasdfme, param in self.modelParams["flow_2"].named_parameters():
            reg = reg + (param ** 2).sum()
        return  -lam * reg
    
    def model(self,data):
        ### this defines your unnormalized posterior
        ### priors
        theta = pyro.sample(
            "theta",
            dist.MultivariateNormal(
                loc=torch.zeros(4),
                covariance_matrix=self.trueParams['priorVar'] * torch.eye(4)
            )
        )
        # mu1 = theta[:2]
        # mu2 = theta[2:]
        
        #mix = dist.Categorical(self.trueParams["weights"])
        
        ### likelihood
        comp = dist.MultivariateNormal(
            loc=theta,#torch.stack([mu1, mu2], dim=0),  # shape (2, 2)
            covariance_matrix=self.trueParams["covs"]     # shape (2, 2, 2)
        ) 
        likl = comp# dist.MixtureSameFamily(mix, comp)
        # tell pyro that all data is independent but samples the same parameter, the likelihood
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", likl, obs=data) #gmm(theta)

    def guide(self,data):
        #here we define the variational approximation
        #Two marginal flows
        pyro.module("flow_1", self.modelParams['flow_1'])
        pyro.module("flow_2", self.modelParams['flow_2'])
        #parmetrization according to the paper
        B    = pyro.param("B", self.modelParams['B'])
        zeta = pyro.param("zeta", self.modelParams["zeta"],constraint=constraints.positive)

        # Regularize
        # reg = self._regularize(lam = 2e-3)  
        # pyro.factor("flow_l2_penalty",reg, has_rsample=True)
        
        params_now = dict(self.modelParams)
        params_now["B"] = B  #force the update
        params_now["zeta"] = zeta
        
        #sample from 4d joint distr, and cast them to the 2 variables
        q     = VectorCopulaFlowQ(params_now) # -> here is the main crux of the implementation
        pyro.sample("theta", q)
        
def get_params_now(pipe):
    # pipe.modelParams contains references to flow modules etc.
    params_now = dict(pipe.modelParams)

    # overwrite anything learned via pyro.param
    params_now["B"]    = pyro.param("B")
    params_now["zeta"] = pyro.param("zeta")
    
    # (optional) if you later move to other pyro.params, add them here too
    return params_now

def per_param_optim_args(param_name):
    # I don't think clipping is necessary stability can be imroved by (but it can do no harm)
    
    # Pyro normalizes names to nn.Module.named_parameters() style
    # when the callable takes one argument.
    print("optimizer saw:", param_name)  #used for debugging
    
    #set LR for different varaibles
    if param_name in {"B", "zeta"}:
        return {"lr": 1e-3, "clip_norm": 5.0}
    elif "flow_1" in param_name or "flow_2" in param_name:
        return {"lr": 1e-3, "clip_norm": 5.0}
    else:
        return {"lr": 1e-3, "clip_norm": 5.0}
    
def runModel(nSteps,printEvery,N,svi,X,pipe):
    step       = 0
    prev_loss  = 0
    diff       = 0
    loss_list = []
    diagnostics = []
    
    while (step < nSteps):
        # print(f"step: {step}")
        loss = svi.step(X)        

        diff = prev_loss - loss
        loss_list.append(loss)
        params_now = get_params_now(pipe)
        
        #make diagnostic plot of loss, split for several terms
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
            PyroParams = {
                "B" : pyro.param("B"),
                "zeta"     : pyro.param("zeta")
                }
            # Inspect parameter store for flow loc/scale parameters
            store = pyro.get_param_store()

            derived    = buildDerivedParams(PyroParams)
            Omega      = derived["Omega"]
            eigvals    = torch.linalg.eigvalsh(Omega)
            now = datetime.now()
            print("")
            print(now)
            print(f"step: {step}, diff: {diff}")
            print(f"loss: {loss}, loss/N: {loss / N}")
            # print(f"Omega: {Omega}")
            # print(f"eigvals: {eigvals}")
            
            print("\n--- Pyro param store ---")
            for name in sorted(store.keys()):
                if name not in ['B','zeta']:
                    val = store[name].detach()
                    print(f"{name}")
                    print(val)
                
            print("\n--- Pyro marginals of estimated Q ---")
            
            # marginals
            params_now = dict(pipe.modelParams)
            params_now["B"] = pyro.param("B")  #force the update
            params_now["zeta"] = pyro.param("zeta")
            q = VectorCopulaFlowQ(params_now)

            samples = q.sample((5000,))   # (5000, 4)
            compare_moments(samples) #compare empirical and true posterior distributions as final test
            
        prev_loss = loss
        step +=1
    return loss_list,diagnostics
    
def postProcessTest(loss_list,diagnostics,pipe,X):
    # ---- plot diagnostic curve ----
    plt.figure()
    plt.plot(np.log(loss_list))
    plt.xlabel("step")
    plt.ylabel("log loss")
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
    ax1.legend(loc="lower left")
    ax1.grid(True)
    plt.title("SVI Diagnostics: Marginals, Copula, and Loss")
    plt.tight_layout()
    plt.savefig("toy_problem/"+filename, dpi=300, bbox_inches="tight")

    #test 1 log_prob
    params_now = dict(pipe.modelParams)
    params_now["B"] = pyro.param("B")  #force the update
    params_now["zeta"] = pyro.param("zeta")
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
    params_now["B"] = pyro.param("B")  #force the update
    params_now["zeta"] = pyro.param("zeta")
    q = VectorCopulaFlowQ(params_now)

    samples = q.sample((5000,))   # (5000, 4)

    mu1_samples = samples[:, 0:2]
    mu2_samples = samples[:, 2:4]

    plotMarg(mu1_samples.T, filename="toy_problem/mu1_samples")
    plotMarg(mu2_samples.T, filename="toy_problem/mu2_samples")
    plotMarg(X.T, filename="toy_problem/data")
    
    #4 compare Posterior:
    compare_moments(samples)
    
def getPosteriorSigma(N):
    trueParams = getTrueParams()
    Sigma  = trueParams['covs']
    Sigma0 = trueParams['priorVar'] * torch.eye(4)
    return torch.linalg.inv(N*torch.linalg.inv(Sigma)+ torch.linalg.inv(Sigma0))

def compare_moments(samples: torch.Tensor,muTrue: torch.Tensor = torch.zeros(4),N_data = 2048):
    """
    samples: (N, d)
    mu:      (d,)
    Sigma:   (d, d)
    """
    N = samples.shape[0]

    # Empirical mean
    mu_hat = samples.mean(dim=0)  # (d,)

    # Centered samples
    X_centered = samples - mu_hat  # (N, d)

    # Empirical covariance (unbiased: divide by N-1)
    Sigma_hat = (X_centered.T @ X_centered) / (N - 1)  # (d, d)

    sigmaTrue = getPosteriorSigma(N_data)
    # Differences
    mean_diff = mu_hat - muTrue
    cov_diff  = Sigma_hat - sigmaTrue
    
    mean_l2 = torch.norm(mean_diff, p=2)       # Euclidean norm
    cov_fro = torch.norm(cov_diff, p='fro')    #

    # print(f"mu_hat: {mu_hat}")
    print(f"Sigma_hat: {Sigma_hat}")
    # print(f"muTrue: {muTrue}")
    print(f"sigmaTrue: {sigmaTrue}")
    # print(f"mean_diff: {mean_diff}")
    # print(f"cov_diff: {cov_diff}")
    
    # print(f"||mean_diff||_2 (Euclidean): {mean_l2}")
    # print(f"||cov_diff||_F (Frobenius): {cov_fro}")

def main():
    pyro.clear_param_store()
    N          = 2048
    batch_size = 512 #512 2048 128
    nSteps     = 2000
    printEvery = 100
    pipe       = pyroCopulaSVIPipeline(batch_size=batch_size,useIdentity = False)
    
    #standardize the data
    X           = sample_true_data(N, pipe.trueParams)
    # mu         = X.mean(0)
    # std        = X.std(0)
    # X          = (X - mu) / std
    
    plot_posterior_marginals()  #calculates theoretical posterior (for our model) and plots it (to be used for comparison)
    loss      = Trace_ELBO(num_particles=10) #pyro machinery
    #for each param per_param_optim_args is called and args are given 
    # -> allow for per parameter setting so different learning rates per param (which adam does implicitely)
    # -> tried didn't work
    optimizer = pyro.optim.ClippedAdam(per_param_optim_args) 
    
    svi       = SVI(pipe.model, pipe.guide, optimizer, loss=loss)
    
    #main part of the model of the model
    loss_list,diagnostics = runModel(nSteps,printEvery,N,svi,X,pipe)
    postProcessTest(loss_list,diagnostics,pipe,X)
    
if __name__ == "__main__":
    main() #weight decay because of the depth of the network?
    # ModelTest = VectorCopulaFlowQ(get_params())
    # result = ModelTest.testInternalConsistency(100000,100000)
    print("end")
    #TODO: fix model -> sample from ML model