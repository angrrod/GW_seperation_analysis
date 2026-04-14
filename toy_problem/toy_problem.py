import torch
from torch.distributions import MultivariateNormal
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
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from pyro.contrib.zuko import ZukoToPyro
from torch.distributions import Transform, constraints
from zuko.lazy import Flow, LazyTransform, UnconditionalDistribution
from zuko.distributions import DiagNormal
from abc import ABC,abstractmethod

class FullAffineTransform(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    sign = +1

    def __init__(self, loc: torch.Tensor, scale_tril: torch.Tensor, cache_size: int = 0):
        super().__init__(cache_size=cache_size)
        self.loc = loc
        self.scale_tril = scale_tril

    @property
    def event_dim(self):
        return 1

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Data -> base:
            z = L^{-1} (x - loc)
        """
        rhs = x - self.loc
        if rhs.ndim == 1:
            return torch.linalg.solve_triangular(
                self.scale_tril,
                rhs.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
        else:
            return torch.linalg.solve_triangular(
                self.scale_tril,
                rhs.transpose(-1, -2),
                upper=False,
            ).transpose(-1, -2)

    def _inverse(self, z: torch.Tensor) -> torch.Tensor:
        """
        Base -> data:
            x = loc + L z
        """
        return z @ self.scale_tril.mT + self.loc

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y = _call(x) = L^{-1}(x-loc), so
        log|det dy/dx| = -log|det L|
        """
        diag = torch.diagonal(self.scale_tril, dim1=-2, dim2=-1)
        ladj = -torch.sum(torch.log(diag))
        return ladj.expand(x.shape[:-1])
class LazyFullAffineTransform(LazyTransform):
    """
    A true Zuko lazy transform:
      z -> x = mu + L z
    where L is lower triangular with positive diagonal.
    """

    def __init__(
        self,
        features: int,
        loc_init: torch.Tensor | None = None,
        scale_tril_init: torch.Tensor | None = None,
        eps: float = 1e-4,
    ):
        super().__init__()

        self.features = features
        self.eps = eps

        if loc_init is None:
            loc_init = torch.zeros(features)

        if scale_tril_init is None:
            scale_tril_init = torch.eye(features)

        if loc_init.shape != (features,):
            raise ValueError(f"loc_init must have shape ({features},), got {tuple(loc_init.shape)}")
        if scale_tril_init.shape != (features, features):
            raise ValueError(
                f"scale_tril_init must have shape ({features}, {features}), "
                f"got {tuple(scale_tril_init.shape)}"
            )

        self.loc = nn.Parameter(loc_init.clone())

        strict_lower = torch.tril(scale_tril_init.clone(), diagonal=-1)
        diag0 = torch.diagonal(scale_tril_init, dim1=-2, dim2=-1)

        # inverse softplus so that softplus(raw_diag)+eps = diag0 at init
        raw_diag0 = torch.log(torch.expm1(torch.clamp(diag0 - eps, min=1e-8)))

        raw_L0 = strict_lower + torch.diag(raw_diag0)
        self.raw_L = nn.Parameter(raw_L0)

    def scale_tril(self) -> torch.Tensor:
        strict_lower = torch.tril(self.raw_L, diagonal=-1)
        raw_diag = torch.diagonal(self.raw_L, dim1=-2, dim2=-1)
        pos_diag = F.softplus(raw_diag) + self.eps
        return strict_lower + torch.diag(pos_diag)

    def forward(self, c: torch.Tensor | None = None) -> Transform:
        # c is ignored because this is unconditional
        return FullAffineTransform(
            loc=self.loc,
            scale_tril=self.scale_tril(),
        )

def make_full_affine_zuko_flow(
    features: int,
    loc_init: torch.Tensor | None = None,
    scale_tril_init: torch.Tensor | None = None,
):
    return Flow(
        transform=LazyFullAffineTransform(
            features=features,
            loc_init=loc_init,
            scale_tril_init=scale_tril_init,
        ),
        base=UnconditionalDistribution(
            DiagNormal,
            torch.zeros(features),
            torch.ones(features),
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
        [1.0, 0.6, 0.0, 0.0],
        [0.6, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.45],
        [0.0, 0.0, 0.45, 1.0],
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

def getModelParams():
    #prior and likl models for pyro
    params = {
        "means" : torch.tensor([0.0,0.0,0.0,0.0]),
        "cov" : torch.tensor([
            [1.0, 0.6, 0.2, 0.1],
            [0.6, 1.0, 0.05, 0.5],
            [0.2, 0.05, 1.0, 0.45],
            [0.1, 0.5, 0.45, 1.0],
        ]),
        "priorVar" : 20,
        "observedVar" : 1
    }
    return params

def ScaleSigma(Sigma,rho = 0.9999):
    eps = 1e-6
    smax = torch.linalg.svdvals(Sigma).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    return Sigma* rho * (1.0 - eps) / (scale + eps)

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

def plot_posterior_marginals(pipe,X,save_dir="toy_problem", filename="true_marginals.png"):

    os.makedirs(save_dir, exist_ok=True)
    mu = torch.zeros(4)
    mu_true, Sigma_true = getPosteriorTarget(
        X,
        priorVar=pipe.ModelParams["priorVar"],
        observedVar=pipe.ModelParams["observedVar"],
    )
    dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma_true)
    X   = dist.sample((5000,))
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
    def __init__(self,flow_1,flow_2,B,zeta,isIndependentCopula,useGaussianBase,useIdentityTransform):
        self.flow_1 = flow_1
        self.flow_2 = flow_2
        self.B      = B
        self.zeta   = zeta
        batch_shape = torch.Size()
        
        #Debug switches
        self.isIndependentCopula  = isIndependentCopula
        self.useGaussianBase      = useGaussianBase 
        self.useIdentityTransform = useIdentityTransform 
        
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([4]), validate_args=None)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        total,_,_,_,_,_ = self.logProbVectorCopula(value)
        return total
    
    def rsample(self, sample_shape=torch.Size()):
        if len(sample_shape) == 0:
            return self._sampleVectorCopulaModel(
                1
            ).squeeze(0)

        N = math.prod(sample_shape)
        
        return self._sampleVectorCopulaModel(
            N
        )
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def _buildOmega(self):
        # build derived matrix
        OmegaBar        = self.zeta*torch.eye(4) + self.B @ self.B.T
        dimList         = [2,2]
        L               = torch.linalg.cholesky(Blockdiag(OmegaBar,dimList))
        I               = torch.eye(L.shape[-1], device=L.device, dtype=L.dtype)
        A               = torch.linalg.solve_triangular(L, I, upper=False)
        Omega           = A @ OmegaBar @ A.T
        # Omega           = ScaleSigma(Omega)
        #matrix will be pos definite if shur compliment I - A^T*A is positive definite which translates to |v|^2 > |Av|^2 for all v and is the case if the largest eigenvalue is smaller than 1 or the spectral norm is smaller than 1
        return Omega

    def logProbVectorCopula(self,theta):
        #allow for batch processing or single event:
        if len(theta.shape) == 2:
            theta02 = theta[:,:2]
            theta24 = theta[:,2:]
        else:
            theta02 = theta[:2]
            theta24 = theta[2:]
            
        # stdn = torch.distributions.Normal( #-> cast inverse to get better precision
        #     torch.tensor(0.),
        #     torch.tensor(1.),
        # )
        
        #get log prob of the marginal flows using made products
        if self.useIdentityTransform:
            logp_marg_1 = torch.tensor(0)
            logp_marg_2 = torch.tensor(0)
            #generate input for log prob copula
            Q_1 = theta02
            Q_2 = theta24
        else:
            dist_1 = self.flow_1()
            dist_2 = self.flow_2()
            logp_marg_1 = dist_1.log_prob(theta02)
            logp_marg_2 = dist_2.log_prob(theta24)
            #generate input for log prob copula
            Q_1 = dist_1.transform.inv(theta02)
            Q_2 = dist_2.transform.inv(theta24)
            
        if len(theta.shape) == 2:
            dim = 1
        else: #len = 1
            dim = 0
        Q = torch.concat([Q_1,Q_2],dim = dim)
        
        logDensity,logDetTerm,logCopulaTerm  = self._logProbCopula(Q)
        total = logp_marg_1 + logp_marg_2 + logDensity
        return total,logp_marg_1,logp_marg_2,logDensity,logDetTerm,logCopulaTerm
    
    def _sampleGaussianBase(self,N):
        if self.isIndependentCopula:
            Omega = torch.eye(4)
        else:
            Omega = self._buildOmega()
        dist = torch.distributions.MultivariateNormal(loc=torch.tensor([0.,0.,0.,0.])
                                                    , covariance_matrix=Omega)
        sample = dist.rsample((N,))
        return sample

    def _sampleVectorCopula(self,N):
        res  = self._sampleGaussianBase(N)  # replace with multivariate distr
        stdn = torch.distributions.Normal(
            torch.tensor(0.),
            torch.tensor(1.),
        )
        if self.useGaussianBase: #gaussian base says to use \phi(\phi^-1)) 
            Z    = stable_unit_interval(stdn.cdf(res))  
        else:
            Z    = res
        return Z[:,0:2],Z[:,2:4]

    def _sampleVectorCopulaModel(self,N):
        dist_1 = self.flow_1()
        dist_2 = self.flow_2()
        if self.isIndependentCopula:
            sample_1 = dist_1.rsample((N,))
            sample_2 = dist_2.rsample((N,))
        else:
            Z_1,Z_2 = self._sampleVectorCopula(N)
            stdn = torch.distributions.Normal(
                torch.tensor(0.),
                torch.tensor(1.),
            )
            
            if self.useIdentityTransform:
                if self.useGaussianBase:
                    raise ValueError("when using the identity we cannot use a gaussian base")
                sample_1 = Z_1
                sample_2 = Z_2
            else:
                if self.useGaussianBase:
                    sample_1 = dist_1.transform(stdn.icdf(Z_1)) #can be used because they are independent
                    sample_2 = dist_2.transform(stdn.icdf(Z_2))
                else:
                    sample_1 = dist_1.transform(Z_1)
                    sample_2 = dist_2.transform(Z_2)
        return torch.cat([sample_1,sample_2], dim=1)

    def _logProbCopula(self,Q,d = 4):
        if self.isIndependentCopula:
            return torch.tensor(0),torch.tensor(0),torch.tensor(0)
        Omega           = self._buildOmega()
        I               = torch.eye(4)
        _, logabsdet    = torch.linalg.slogdet(Omega)
        OmegaInv        = torch.linalg.solve(Omega, I) 
        PhiInv          = Q
        if len(Q.shape) == 2:
            einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (OmegaInv- I), PhiInv)  #sum in order to deal with quadratic form dimensions
        else: #len = 1 so one event
            einsum = PhiInv.T @ (OmegaInv- I) @ PhiInv
        logDetTerm      = (-1/2)*logabsdet
        logCopulaTerm   = (-1/2)*einsum
        logDensity      = logDetTerm + logCopulaTerm #for exact expression, constant is needed-> not used
        return logDensity,logDetTerm,logCopulaTerm

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
    
def _empirical_mean_cov(x: torch.Tensor):
    mu = x.mean(dim=0)
    xc = x - mu
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    return mu, cov

def _softplus_inverse(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(x, min=eps)
    return torch.log(torch.expm1(x))

def set_full_affine_flow_params(flow_module, loc: torch.Tensor, scale_tril: torch.Tensor, eps: float = 1e-4):
    """
    Directly set parameters of make_full_affine_zuko_flow(features=2)
    so that the resulting affine transform is exactly:
        x = loc + L z
    """
    lazy = flow_module.transform  # LazyFullAffineTransform

    with torch.no_grad():
        lazy.loc.copy_(loc)

        strict_lower = torch.tril(scale_tril, diagonal=-1)
        diag = torch.diagonal(scale_tril, dim1=-2, dim2=-1)
        raw_diag = _softplus_inverse(diag - eps)
        raw_L = strict_lower + torch.diag(raw_diag)

        lazy.raw_L.copy_(raw_L)

def test_independent_sampling_matches_block_structure(pipe, n_test: int = 20000):
    """
    Check whether q.sample() has the same empirical moments as sampling
    independently from flow_1 and flow_2 and concatenating.
    """
    if not pipe.isIndependentCopula:
        raise ValueError("This test is only intended for isIndependentCopula=True.")

    with torch.no_grad():
        q = pipe.make_q()
        dist_1 = pipe.flow_1()
        dist_2 = pipe.flow_2()

        x_q = q.sample((n_test,))
        x_manual = torch.cat(
            [dist_1.rsample((n_test,)), dist_2.rsample((n_test,))],
            dim=1,
        )

        mu_q, cov_q = _empirical_mean_cov(x_q)
        mu_manual, cov_manual = _empirical_mean_cov(x_manual)

        mean_err = torch.norm(mu_q - mu_manual, p=2).item()
        cov_err = torch.norm(cov_q - cov_manual, p="fro").item()

        cross_block_q = cov_q[:2, 2:]
        cross_block_manual = cov_manual[:2, 2:]

        print("\n=== test_independent_sampling_matches_block_structure ===")
        print("||mu_q - mu_manual||_2:", mean_err)
        print("||cov_q - cov_manual||_F:", cov_err)
        print("max |cross_block_q|:", cross_block_q.abs().max().item())
        print("max |cross_block_manual|:", cross_block_manual.abs().max().item())

        return {
            "x_q": x_q,
            "x_manual": x_manual,
            "mu_q": mu_q,
            "cov_q": cov_q,
            "mu_manual": mu_manual,
            "cov_manual": cov_manual,
            "mean_l2": mean_err,
            "cov_fro": cov_err,
        }

def test_independent_log_prob_matches_manual(pipe, n_test: int = 2048, atol: float = 1e-6):
    """
    In the independent-copula case, q.log_prob(theta) should equal
    dist_1.log_prob(theta[:,:2]) + dist_2.log_prob(theta[:,2:]).
    """
    if not pipe.isIndependentCopula:
        raise ValueError("This test is only intended for isIndependentCopula=True.")

    with torch.no_grad():
        q = pipe.make_q()
        theta = q.sample((n_test,))

        dist_1 = pipe.flow_1()
        dist_2 = pipe.flow_2()

        lp_q = q.log_prob(theta)
        lp_manual = dist_1.log_prob(theta[:, :2]) + dist_2.log_prob(theta[:, 2:])

        abs_err = (lp_q - lp_manual).abs()
        max_err = abs_err.max().item()
        mean_err = abs_err.mean().item()

        print("\n=== test_independent_log_prob_matches_manual ===")
        print("max |lp_q - lp_manual|:", max_err)
        print("mean |lp_q - lp_manual|:", mean_err)
        print("allclose:", torch.allclose(lp_q, lp_manual, atol=atol, rtol=1e-5))

        return {
            "theta": theta,
            "lp_q": lp_q,
            "lp_manual": lp_manual,
            "max_abs_err": max_err,
            "mean_abs_err": mean_err,
        }

def make_reference_block_guide_module():
    return ReferenceBlockDiagonalGuide()

def reference_block_guide_fn(ref_guide_module):
    def _guide(data):
        pyro.module("ref_block_guide", ref_guide_module)
        pyro.sample("theta", ref_guide_module.dist())
    return _guide

def test_reference_block_guide_svi(X: torch.Tensor, pipe, n_steps: int = 3000,printEvery = 50):
    """
    Run SVI with a plain block-diagonal Gaussian guide that matches the same
    family as your custom independent 2x2 guide, but without VectorCopulaFlowQ.
    """
    pyro.clear_param_store()

    ref_module = make_reference_block_guide_module()
    ref_guide = reference_block_guide_fn(ref_module)

    optimizer = pyro.optim.ClippedAdam({"lr": 2e-4, "clip_norm": 5.0})
    loss = Trace_ELBO(num_particles=50)
    svi = SVI(pipe.model, ref_guide, optimizer, loss=loss)

    losses = []
    for step in range(n_steps):
        if step % printEvery == 0:
            print(f'step: {step}')
        loss_val = svi.step(X)
        losses.append(loss_val)

    with torch.no_grad():
        samples = []
        for _ in range(2000):
            tr = pyro.poutine.trace(ref_guide).get_trace(X)
            samples.append(tr.nodes["theta"]["value"].detach())
        samples = torch.stack(samples, dim=0)

        moment_result = compare_moments(samples, pipe, X)

    print("\n=== test_reference_block_guide_svi ===")
    print("tail mean loss:", (np.array(losses[-500:]) / X.shape[0]).mean())
    print("tail std loss:", (np.array(losses[-500:]) / X.shape[0]).std())

    return {
        "losses": losses,
        "samples": samples,
        "moment_result": moment_result,
    }
class ReferenceBlockDiagonalGuide(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.loc1 = nn.Parameter(torch.zeros(2))
        self.raw_L1 = nn.Parameter(torch.eye(2))
        self.loc2 = nn.Parameter(torch.zeros(2))
        self.raw_L2 = nn.Parameter(torch.eye(2))
        self.eps = eps

    def _scale_tril(self, raw_L: torch.Tensor):
        strict_lower = torch.tril(raw_L, diagonal=-1)
        raw_diag = torch.diagonal(raw_L, dim1=-2, dim2=-1)
        pos_diag = F.softplus(raw_diag) + self.eps
        return strict_lower + torch.diag(pos_diag)

    def dist(self):
        L1 = self._scale_tril(self.raw_L1)
        L2 = self._scale_tril(self.raw_L2)

        loc = torch.cat([self.loc1, self.loc2])

        cov = torch.zeros(4, 4, dtype=loc.dtype, device=loc.device)
        cov[:2, :2] = L1 @ L1.T
        cov[2:, 2:] = L2 @ L2.T

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)

def test_custom_q_against_known_block_gaussian(pipe, n_test: int = 4096):
    """
    Force flow_1 and flow_2 to represent a known block-diagonal Gaussian,
    then compare q.sample() and q.log_prob() against the exact 4D MVN.
    """
    if not pipe.isIndependentCopula:
        raise ValueError("This test is only intended for isIndependentCopula=True.")

    # choose an explicit block-diagonal Gaussian target
    mu1 = torch.tensor([0.3, -0.2])
    mu2 = torch.tensor([0.5,  0.1])

    L1 = torch.tensor([[0.7, 0.0],
                       [0.2, 0.5]])
    L2 = torch.tensor([[0.4, 0.0],
                       [-0.1, 0.6]])

    Sigma1 = L1 @ L1.T
    Sigma2 = L2 @ L2.T

    mu = torch.cat([mu1, mu2])
    Sigma = torch.zeros(4, 4)
    Sigma[:2, :2] = Sigma1
    Sigma[2:, 2:] = Sigma2

    # overwrite the flow parameters
    set_full_affine_flow_params(pipe.flow_1, mu1, L1)
    set_full_affine_flow_params(pipe.flow_2, mu2, L2)

    with torch.no_grad():
        q = pipe.make_q()
        ref = torch.distributions.MultivariateNormal(mu, covariance_matrix=Sigma)

        # compare log_prob on common test points
        x_test = ref.sample((n_test,))
        lp_q = q.log_prob(x_test)
        lp_ref = ref.log_prob(x_test)

        lp_abs_err = (lp_q - lp_ref).abs()

        # compare sampled moments
        x_q = q.sample((20000,))
        mu_q, cov_q = _empirical_mean_cov(x_q)

        mean_err = torch.norm(mu_q - mu, p=2).item()
        cov_err = torch.norm(cov_q - Sigma, p="fro").item()

        print("\n=== test_custom_q_against_known_block_gaussian ===")
        print("max |lp_q - lp_ref|:", lp_abs_err.max().item())
        print("mean |lp_q - lp_ref|:", lp_abs_err.mean().item())
        print("||mu_q - mu||_2:", mean_err)
        print("||cov_q - Sigma||_F:", cov_err)

        return {
            "mu_target": mu,
            "Sigma_target": Sigma,
            "mu_q": mu_q,
            "cov_q": cov_q,
            "lp_q": lp_q,
            "lp_ref": lp_ref,
            "lp_max_abs_err": lp_abs_err.max().item(),
            "lp_mean_abs_err": lp_abs_err.mean().item(),
            "mean_l2": mean_err,
            "cov_fro": cov_err,
        }

# perform SVI optimization

class pyroPipe(ABC):
    def __init__(self,X,num_particles,batch_size):
        self.ModelParams  = getModelParams()
        self.batch_size   = batch_size
        self.optimizer    = pyro.optim.ClippedAdam(self._per_param_optim_args) 
        self.loss         = Trace_ELBO(num_particles=num_particles) #pyro machinery
        self.X            = X
        N,_               = X.shape
        self.N            = N
        self.has_q        = False
        
    @abstractmethod
    def model(self):
        raise NotImplementedError
    
    @abstractmethod
    def guide(self):
        raise NotImplementedError
    
    @abstractmethod
    def getOptimalState(self):
        raise NotImplementedError
    
    def trainModel(self,nSteps,printEvery):
        step        = 0
        prev_loss   = 0
        diff        = 0
        loss_list   = []
        diagnostics = []
        nSteps      = nSteps
        printEvery  = printEvery
        svi         = SVI(self.model, self.guide,  self.optimizer, loss=self.loss)
        
        for step in range(nSteps):
            
            loss = svi.step(self.X)        
            diff = prev_loss - loss
            loss_list.append(loss)
            
            # -- make diagnostic plot of loss, split for several terms --
            with torch.no_grad():
                if self.has_q:
                    q = self.make_q()
                    guide_trace = pyro.poutine.trace(self.guide).get_trace(self.X)
                    theta_sample = guide_trace.nodes["theta"]["value"].detach()
                    total, m1, m2, cop,logDetTerm,logCopulaTerm = q.logProbVectorCopula(theta_sample)
                    diag = dict(
                        total  = total.mean().item(),
                        marg1  = m1.mean().item(),
                        marg2  = m2.mean().item(),
                        # copula = cop.mean().item(),
                        # det    = logDetTerm.mean().item(),
                        # gauss  = logCopulaTerm.mean().item(),
                    )
                    diagnostics.append(diag)
                
            if step % printEvery == 0:           
                # PyroParams = {
                #     "B" : pyro.param("B"),
                #     "zeta"     : pyro.param("zeta")
                #     }
                # # Inspect parameter store for flow loc/scale parameters
                store = pyro.get_param_store()

                # derived    = buildOmega(PyroParams)
                # Omega      = derived["Omega"]
                # eigvals    = torch.linalg.eigvalsh(Omega)
                
                now = datetime.now()
                print("")
                print(now)
                print(f"step: {step}, diff: {diff}")
                print(f"loss: {loss}, loss/N: {loss / self.N}")
                # print(f"Omega: {Omega}")
                # print(f"eigvals: {eigvals}")
                
                # print("\n--- Pyro param store ---")
                # for name in sorted(store.keys()):
                #     if name not in ['B','zeta']:
                #         val = store[name].detach()
                #         print(f"{name}")
                #         print(val)
                    
                # print("\n--- Pyro marginals of estimated Q ---")
                
                # marginals
                if self.has_q:
                    q = self.make_q()
                    samples = q.sample((5000,))   # (5000, 4)
                
            prev_loss = loss
            step +=1
        
        return loss_list,diagnostics,self.getOptimalState()

    def _per_param_optim_args(self,param_name):
        # I don't think clipping is necessary stability can be imroved by (but it can do no harm)
        
        # Pyro normalizes names to nn.Module.named_parameters() style
        # when the callable takes one argument.
        print("optimizer saw:", param_name)  #used for debugging
        
        #set LR for different varaibles
        # if param_name in {"B", "zeta"}:
        #     return {"lr": 2e-4, "clip_norm": 5.0}
        # elif "flow_1" in param_name or "flow_2" in param_name:
        #     return {"lr": 2e-4, "clip_norm": 5.0}
        # else:
        return {"lr": 5e-4, "clip_norm": 5.0}

class pyroCopulaSVIPipeline(pyroPipe):
    def __init__(self,X,batch_size,num_particles, flow_1_chkpt = None, flow_2_chkpt = None):
        super().__init__(X,num_particles,batch_size)
        # Debug switches
        self.isIndependentCopula  = True #do we use the independence copula
        self.useGaussianBase      = False #if True we assume we cannot take a numerical shortcut
        self.useIdentityTransform = False #Marginal flows used is the special identity transform
        
        self.has_q        = True
        
        if self.useIdentityTransform:
            self.flow_1 = make_identity_flow_module_2d()
            self.flow_2 = make_identity_flow_module_2d()
        else:
            self.flow_1 = zuko.flows.NSF(
                features=2,
                transforms=2,
                context=0,
                hidden_features=(8,),
                bins=8,
                randperm=False,
            )
            
            self.flow_2 = zuko.flows.NSF(
                features=2,
                transforms=2,
                context=0,
                hidden_features=(8,),
                bins=8,
                randperm=False,
            )
            # self.flow_1 = make_full_affine_zuko_flow(features=2) 
            # self.flow_2 = make_full_affine_zuko_flow(features=2) 
            
        self.B_init = torch.eye(4)#torch.tensor([
        #     [0.2, 0.4, 0.0, 1.0],
        #     [0.8, 0.1, 0.2, 0.3],
        #     [1.2, 0.4, 1.0, 0.0],
        #     [1.4, 0.0, 0.1, 1.0]
        # ])
        self.zeta_init = torch.tensor(1.0)
        self.flow_1.load_state_dict(flow_1_chkpt)
        self.flow_2.load_state_dict(flow_2_chkpt)
    
    def model(self,data):
        ### this defines your unnormalized posterior
        ### priors
        theta = pyro.sample(
            "theta",
            dist.MultivariateNormal(
                loc=torch.zeros(4),
                covariance_matrix = (self.ModelParams['priorVar'] ** 2) * torch.eye(4)
            )
        )
        
        ### likelihood
        likl = dist.MultivariateNormal(
            loc=theta,
            covariance_matrix = (self.ModelParams["observedVar"] ** 2)*torch.eye(4)     # shape (2, 2, 2)
        ) 

        # tell pyro that all data is independent but samples the same parameter, the likelihood
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", likl, obs=data) #gmm(theta)

    def guide(self, data):
        # pyro.module("flow", self.flow)
        q = self.make_q() #ZukoToPyro(self.flow())          # instantiate lazy flow -> actual distribution
        pyro.sample("theta", q)
        
    def make_q(self):
        pyro.module("flow_1", self.flow_1)
        pyro.module("flow_2", self.flow_2)

        if self.isIndependentCopula:
            B    = None
            zeta = None
        else:
            B    = pyro.param("B", self.B_init) #B = to existing parameter in param store or if it is not present, self.B_init
            zeta = pyro.param("zeta", self.zeta_init, constraint=constraints.positive)

        q = VectorCopulaFlowQ(
            flow_1               = self.flow_1,
            flow_2               = self.flow_2,
            B                    = B,
            zeta                 = zeta,
            isIndependentCopula  = self.isIndependentCopula,
            useGaussianBase      = self.useGaussianBase,
            useIdentityTransform = self.useIdentityTransform
        )
        return q
    
    def guide_direct_nsf(self, data):
        pyro.module("flow_1", self.flow_1)
        pyro.module("flow_2", self.flow_2)

        d1 = self.flow_1()
        d2 = self.flow_2()

        class ProductQ(dist.TorchDistribution):
            support = constraints.real_vector
            has_rsample = True

            def __init__(self):
                super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([4]))

            def rsample(self, sample_shape=torch.Size()):
                x1 = d1.rsample(sample_shape)
                x2 = d2.rsample(sample_shape)
                return torch.cat([x1, x2], dim=-1)

            def log_prob(self, value):
                return d1.log_prob(value[..., :2]) + d2.log_prob(value[..., 2:])
        
        pyro.sample("theta", ProductQ())

    def getOptimalState(self):
        return None

class pyroMarginalSVIPipeline(pyroPipe): 
    def __init__(self,X,num_particles,batch_size = 512,marginal_dim = 2):
        super().__init__(X,num_particles,batch_size)
        self.marginal_dim = marginal_dim
        self.flow = zuko.flows.NSF(
            features=2,
            transforms=2,
            context=0,
            hidden_features=(8,),
            bins=8,
            randperm=False,
        )
        # self.flow         = make_full_affine_zuko_flow(features=self.marginal_dim)
        self.ModelParams  = getModelParams()
        
    def model(self,data):
        ### this defines your unnormalized posterior
        ### priors
        theta = pyro.sample(
            "theta",
            dist.MultivariateNormal(
                loc               = torch.zeros(self.marginal_dim),
                covariance_matrix = (self.ModelParams['priorVar'] ** 2) * torch.eye(self.marginal_dim)
            )
        )
        
        ### likelihood
        likl = dist.MultivariateNormal(
            loc               = theta,
            covariance_matrix = (self.ModelParams["observedVar"] ** 2)*torch.eye(self.marginal_dim)     # shape (2, 2, 2)
        ) 

        # tell pyro that all data is independent but samples the same parameter, the likelihood
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", likl, obs=data) #gmm(theta)

    def guide(self, data):
        pyro.module("flow", self.flow)
        q = ZukoToPyro(self.flow())       # instantiate lazy flow -> actual distribution
        pyro.sample("theta", q)

    def getOptimalState(self):
        return self.flow.state_dict()

def plotLoss(fileName:str,loss_list):
    plt.figure()
    plt.plot(np.log(loss_list))
    plt.xlabel("step")
    plt.ylabel("log loss")
    plt.title("SVI loss difference diagnostic")

    plt.savefig("toy_problem/"+fileName, dpi=300, bbox_inches="tight")

def postProcessTest(loss_list,diagnostics,pipe,X):
    N,_ = X.shape
    loss_arr = np.array(loss_list) / N   # normalize per datapoint if you want
    tail = loss_arr[-500:]  # last 500 steps
    
    print("mean loss:", tail.mean())
    print("std loss:", tail.std())
    
    sigma_p = pipe.ModelParams["priorVar"]
    sigma_o = pipe.ModelParams["observedVar"]

    Sigma_post = 1 / (1/sigma_p**2 + N/sigma_o**2)
    Sigma_post = Sigma_post * torch.eye(4)

    mu_post = (Sigma_post[0,0] / sigma_o**2) * X.sum(dim=0)
    samples = []
    with torch.no_grad():
        for _ in range(2000):
            tr = pyro.poutine.trace(pipe.guide).get_trace(X)
            samples.append(tr.nodes["theta"]["value"].detach())

    samples = torch.stack(samples, dim=0)

    print("guide mean:", samples.mean(dim=0))
    print("guide cov diag:", torch.cov(samples.T).diag())
    
    # ---- plot diagnostic curve ----
    plotLoss("svi_loss_diagnostic.png",loss_list)
    
    ## ---- logprob plots ----
    total  = np.array([d["total"]  for d in diagnostics])
    marg1  = np.array([d["marg1"]  for d in diagnostics])
    marg2  = np.array([d["marg2"]  for d in diagnostics])
    # det    = np.array([d["det"] for d in diagnostics])
    # gauss  = np.array([d["gauss"] for d in diagnostics])
    # copula = np.array([d["copula"] for d in diagnostics])
    steps  = range(len(total))
    filename = "log_prob_sample"
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Left axis → component log-densities
    ax1.plot(steps, marg1, label="marg1", linewidth=2)
    ax1.plot(steps, marg2, label="marg2", linewidth=2)
    # ax1.plot(steps, copula, label="copula", linewidth=2)
    # ax1.plot(steps, det, label="det part of copula", linewidth=2)
    # ax1.plot(steps, gauss, label="gauss part of copula", linewidth=2)
    ax1.plot(steps, total, label="total", linewidth=3, linestyle="--")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Mean log-density components")
    ax1.legend(loc="lower left")
    ax1.grid(True)
    plt.title("SVI Diagnostics: Marginals, Copula, and Loss")
    plt.tight_layout()
    plt.savefig("toy_problem/"+filename, dpi=300, bbox_inches="tight")

    # --- test 1 log_prob ---
    q = pipe.make_q()
    x = q.sample((4096,))
    lp = q.log_prob(x)

    print("finite log_prob on own samples:", torch.isfinite(lp).all().item())
    print("lp mean/std:", lp.mean().item(), lp.std().item())
    
    # #--- test 2 -> correct T use ---
    q = pipe.make_q()

    theta = q.sample((8192,))   # shape (8192, 4)
    mu1 = theta[:, 0:2]
    mu2 = theta[:, 2:4]
    
    dist_1 = pipe.flow_1()
    dist_2 = pipe.flow_2()
    z1 = dist_1.transform.inv(mu1)
    z2 = dist_2.transform.inv(mu2)

    print(f"mu1 shape: {mu1.shape}, mu2 shape: {mu2.shape}")
    print(f"T1^-1(mu1): mean {z1.mean(0)}, var {z1.var(0)}")
    print(f"T2^-1(mu2): mean {z2.mean(0)}, var {z2.var(0)}")
    
    # --- 3 marginals ---
    q = pipe.make_q()

    samples = q.sample((5000,))   # (5000, 4)

    mu1_samples = samples[:, 0:2]
    mu2_samples = samples[:, 2:4]

    plotMarg(mu1_samples.T, filename="toy_problem/mu1_samples")
    plotMarg(mu2_samples.T, filename="toy_problem/mu2_samples")
    plotMarg(X.T, filename="toy_problem/data")
    
    # # --- 4 compare Posterior: ---
    compare_moments(samples,pipe,X)

def getPosteriorTarget(X, priorVar=20.0, observedVar=1.0):
    N, d = X.shape
    sigma_p2 = priorVar**2
    sigma_o2 = observedVar**2

    Sigma_post = (1.0 / (1.0/sigma_p2 + N/sigma_o2)) * torch.eye(d, dtype=X.dtype, device=X.device)
    mu_post = (Sigma_post[0, 0] / sigma_o2) * X.sum(dim=0)
    return mu_post, Sigma_post

def compare_moments(samples: torch.Tensor,pipe,X:torch.tensor):
    """
    samples: (N, d)
    mu:      (d,)
    Sigma:   (d, d)
    """

    # Empirical mean
    mu_hat = samples.mean(dim=0)  # (d,)
    N,_ = samples.shape
    # Centered samples
    X_centered = samples - mu_hat  # (N, d)

    # Empirical covariance (unbiased: divide by N-1)
    Sigma_hat = (X_centered.T @ X_centered) / (N - 1)  # (d, d)

    muTrue, Sigma_true = getPosteriorTarget(
        X,
        priorVar=pipe.ModelParams["priorVar"],
        observedVar=pipe.ModelParams["observedVar"],
    )
    # Differences
    mean_diff = mu_hat - muTrue
    cov_diff  = Sigma_hat - Sigma_true
    
    mean_l2 = torch.norm(mean_diff, p=2)       # Euclidean norm
    cov_fro = torch.norm(cov_diff, p='fro')    #

    # print(f"mu_hat: {mu_hat}")
    print(f"Sigma_hat: {Sigma_hat}")
    # print(f"muTrue: {muTrue}")
    print(f"sigmaTrue: {Sigma_true}")
    # print(f"mean_diff: {mean_diff}")
    # print(f"cov_diff: {cov_diff}")
    
    # print(f"||mean_diff||_2 (Euclidean): {mean_l2}")
    # print(f"||cov_diff||_F (Frobenius): {cov_fro}")

#old debug code
class GaussianGuide(nn.Module):
    def __init__(self, dim=4, eps=1e-4):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(dim))
        self.raw_L = nn.Parameter(0.01 * torch.randn(dim, dim))
        self.eps = eps

    def scale_tril(self):
        L = torch.tril(self.raw_L)
        diag = torch.diagonal(L)
        diag = F.softplus(diag) + self.eps
        L = L - torch.diag(torch.diagonal(L)) + torch.diag(diag)
        return L

    def dist(self):
        return dist.MultivariateNormal(
            loc=self.loc,
            scale_tril=self.scale_tril(),
        )

def main():
    
    #standardize the data
    N             = 2048
    X             = sample_true_data(N, getTrueParams())
    # mu         = X.mean(0)
    # std        = X.std(0)
    # X          = (X - mu) / std
    
    #config
    batch_size    = 512 #512 2048 128
    num_particles = 50
    
    pyro.clear_param_store()
    pipe_init_1  = pyroMarginalSVIPipeline(X[:,0:2], batch_size = batch_size, num_particles = 10,marginal_dim = 2)
    loss_list_1,_,flow_1_chkpt = pipe_init_1.trainModel(1500,100)
    plotLoss("svi_loss_diagnostic_init_1.png",loss_list_1)
    
    pyro.clear_param_store()
    pipe_init_2  = pyroMarginalSVIPipeline(X[:,2:4], batch_size = batch_size, num_particles = 10,marginal_dim = 2)
    loss_list_2,_,flow_2_chkpt = pipe_init_2.trainModel(1500,100)
    plotLoss("svi_loss_diagnostic_init_2.png",loss_list_2)
    
    pyro.clear_param_store()
    pipe = pyroCopulaSVIPipeline(
            X,
            batch_size = batch_size,
            num_particles = num_particles,
            flow_1_chkpt = flow_1_chkpt,
            flow_2_chkpt = flow_2_chkpt
        )
    
    # --- pre-SVI unit tests for custom guide ---
    test_independent_log_prob_matches_manual(pipe)
    test_independent_sampling_matches_block_structure(pipe)
    # test_custom_q_against_known_block_gaussian(pipe)
    
    # rebuild fresh pipeline for actual training
    pipe = pyroCopulaSVIPipeline(
            X,
            batch_size = batch_size,
            num_particles = num_particles,
            flow_1_chkpt = flow_1_chkpt,
            flow_2_chkpt = flow_2_chkpt
        )
    ### calculates theoretical posterior (for our model) 
    ### and plots it (to be used for comparison)
    plot_posterior_marginals(pipe,X)  
    
    # main part of the model of the model
    loss_list,diagnostics,_ = pipe.trainModel(5000,200)
    
    postProcessTest(loss_list,diagnostics,pipe,X)
    
if __name__ == "__main__":
    main()
    # ModelTest = pipe.make_q()
    # result = ModelTest.testInternalConsistency(100000,100000)
    print("end")
    #TODO: fix model -> sample from ML model