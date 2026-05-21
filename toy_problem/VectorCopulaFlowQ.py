
from pyro.distributions.torch_distribution import TorchDistribution
import matplotlib.pyplot as plt
import torch
from torch.distributions import constraints
import math
import numpy as np
from scipy.stats import gaussian_kde
from setUp import getPltDir

def Blockdiag(B,dimList):
    Bdiag = B.clone()
    
    prevDim = 0
    for dim in dimList:
        ind = dim + prevDim
        Bdiag[:ind,ind:] = 0
        Bdiag[ind:,:ind] = 0   
        prevDim += dim  
        
    # print(f"is PSD: {is_psd(Bdiag)}")
    return Bdiag

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

def is_psd(A, tol=1e-8):
    # ensure symmetry first
    if not torch.allclose(A, A.T, atol=tol):
        return False
    
    eigvals = torch.linalg.eigvalsh(A)  # for symmetric/Hermitian matrices
    return torch.all(eigvals >= -tol)

class VectorCopulaFlowQ(TorchDistribution):
    arg_constraints = {}  # fill if you have constrained params
    support         = constraints.real_vector
    has_rsample     = True  # set True if you implement rsample()
    def __init__(self,flow_1,flow_2,B,zeta,isIndependentCopula,useIdentityTransform): 
        self.flow_1 = flow_1
        self.flow_2 = flow_2
        self.B      = B
        self.zeta   = zeta
        batch_shape = torch.Size()
        
        #Debug switches
        self.isIndependentCopula  = isIndependentCopula
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

        N = math.prod(sample_shape) #get one dimensional samples of size equal to the dimensions of the sample
        
        return self._sampleVectorCopulaModel(
            N
        )
    
    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)
        
    def _buildOmega(self):
        eye = torch.eye(4, device=self.B.device, dtype=self.B.dtype)
        OmegaBar = self.zeta * eye + self.B @ self.B.T

        if not torch.isfinite(OmegaBar).all():
            raise RuntimeError("OmegaBar contains NaN/Inf")

        dimList = [2, 2]
        Bd = Blockdiag(OmegaBar, dimList)

        if not torch.isfinite(Bd).all():
            raise RuntimeError("Blockdiag(OmegaBar) contains NaN/Inf")

        Bd = Bd + 1e-6 * eye
        L = torch.linalg.cholesky(Bd)

        A = torch.inverse(L)
        Omega = A @ OmegaBar @ A.T

        if not torch.isfinite(Omega).all():
            raise RuntimeError("Omega contains NaN/Inf")

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
        
        if theta.ndim == 2:
            # batched
            zeros = theta.new_zeros(theta.shape[0])
        else:
            # single event
            zeros = theta.new_tensor(0.0)
        #get log prob of the marginal flows using made products
        if self.useIdentityTransform:
            logp_marg_1 = zeros
            logp_marg_2 = zeros
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
        Z  = self._sampleGaussianBase(N)  # replace with multivariate distr
        return Z[:,0:2],Z[:,2:4]

    def _sampleVectorCopulaModel(self,N):
        dist_1 = self.flow_1()
        dist_2 = self.flow_2()
        if self.isIndependentCopula:
            sample_1 = dist_1.rsample((N,))
            sample_2 = dist_2.rsample((N,))
        else:
            Z_1,Z_2 = self._sampleVectorCopula(N)
            # if self.useIdentityTransform:
            #     sample_1 = torch.randn(N, 2)
            #     sample_2 = torch.randn(N, 2)
            # else:
            sample_1 = dist_1.transform(Z_1)  #numerical shortcut can be removed so no \phi(\phi^-1)) be used because they are independent
            sample_2 = dist_2.transform(Z_2)
        return torch.cat([sample_1,sample_2], dim=1)

    def _logProbCopula(self,Q,d = 4):
        if Q.ndim == 2:
            # batched
            zeros = Q.new_zeros(Q.shape[0])
        else:
            # single event
            zeros = Q.new_tensor(0.0)
        
        if self.isIndependentCopula:
            return zeros,zeros,zeros
        Omega           = self._buildOmega()
        I               = torch.eye(4)
        L = torch.linalg.cholesky(Omega)
        OmegaInv = torch.cholesky_solve(I, L)
        logabsdet = 2 * torch.log(torch.diagonal(L)).sum()
        PhiInv          = Q
        if len(Q.shape) == 2:
            einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (OmegaInv- I), PhiInv)  #sum in order to deal with quadratic form dimensions
        else: #len = 1 so one event
            einsum = PhiInv @ (OmegaInv- I) @ PhiInv.T
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
        pltDir = getPltDir()
        plt.savefig(pltDir+"model_vs_kde.png", dpi=300)
        plot_kde_3d_overlay_from_result(result,filename = 'toy_problem/kde_overaly_3d')
        return result
