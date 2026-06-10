import torch
from torch.distributions import MultivariateNormal
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.constraints import real
import numpy as np
import matplotlib.pyplot as plt
import argparse
from NFs import NF_type
from setUp import getModelParams, getPltDir
from pyroPipes import pyroMarginalSVIPipeline,pyroCopulaSVIPipeline

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
    # mean = torch.tensor([0.0,0.0,0.0,0.0]) 
    mean = torch.tensor([2.0,2.0,2.0,2.0]) 
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
    


    params_true_data = {
            "mean" : mean,
            }
    # components = MultivariateNormal(loc=params_true_data['means'], covariance_matrix=params_true_data['covs'])
    # mixture = MixtureSameFamily(mix, params_true_data['weights'])
    # samp_mix = mixture.sample((10000,))
    # plotMarg(samp_mix.T)
    return params_true_data

def ScaleSigma(Sigma,rho = 0.9999):
    eps = 1e-6
    smax = torch.linalg.svdvals(Sigma).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    return Sigma* rho * (1.0 - eps) / (scale + eps)

def sample_true_data(N:int,true_params,known_cov):
    # z = Categorical(probs=true_params['weights']).sample((N,))          # (N,)
    
    # gather component parameters per datapoint
    mu    = true_params['mean']#[z]                                       # (N, d)
    Sigma = known_cov#[z]                                     # (N, d, d)

    # batch MVN: one MVN per datapoint
    X = MultivariateNormal(loc=mu, covariance_matrix=Sigma).sample((N,))  # (N, d)
    return X

def plot_posterior_marginals(pipe,X, filename="true_marginals.png"):
    mu_true, Sigma_true = getPosteriorTarget(
        X,
        priorVar=pipe.ModelParams["priorVar"],
        cov=pipe.ModelParams["cov"],
    )
    dist = torch.distributions.MultivariateNormal(mu_true, covariance_matrix=Sigma_true)
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
    save_dir = getPltDir()
    path     = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {path}")
    
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

    optimizer = pyro.optim.ClippedAdam({"lr": 1e-3, "clip_norm": 10.0})
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

def plotLoss(fileName:str,loss_list):
    plt.figure()
    plt.plot(np.log(loss_list))
    plt.xlabel("step")
    plt.ylabel("log loss")
    plt.title("SVI loss difference diagnostic")
    pltDir = getPltDir()
    plt.savefig(pltDir+fileName, dpi=300, bbox_inches="tight")

def guideTests(pipe,X):
    samples = []
    with torch.no_grad():
        for _ in range(2000):
            tr = pyro.poutine.trace(pipe.guide).get_trace(X)
            samples.append(tr.nodes["theta"]["value"].detach())

    samples = torch.stack(samples, dim=0)

    print("guide mean:", samples.mean(dim=0))
    print("guide cov diag:", torch.cov(samples.T).diag())
    
def logProbPlot(diagnostics,filename:str):
    total  = np.array([d["total"]  for d in diagnostics])
    marg1  = np.array([d["marg1"]  for d in diagnostics])
    marg2  = np.array([d["marg2"]  for d in diagnostics])
    det    = np.array([d["det"] for d in diagnostics])
    gauss  = np.array([d["gauss"] for d in diagnostics])
    copula = np.array([d["copula"] for d in diagnostics])
    steps  = range(len(total))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Left axis → component log-densities
    ax1.plot(steps, marg1, label="marg1", linewidth=2)
    ax1.plot(steps, marg2, label="marg2", linewidth=2)
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
    pltDir = getPltDir()
    plt.savefig(pltDir+filename, dpi=300, bbox_inches="tight")
    
def postProcessTest(loss_list,diagnostics,pipe,X,run:int):
    N,_ = X.shape
    loss_arr = np.array(loss_list) / N   # normalize per datapoint if you want
    tail = loss_arr[-500:]  # last 500 steps
    
    print("mean loss:", tail.mean())
    print("std loss:", tail.std())
    
    guideTests(pipe,X)
    
    # ---- plot diagnostic curve ----
    plotLoss(f"svi_loss_diagnostic_{run}.png",loss_list)
    
    # ---- logprob plots ----
    if pipe.has_q:
        logProbPlot(diagnostics,filename = f"log_prob_sample_{run}.png")
    # ---- other tests ----
    if pipe.has_q:
        makeTests(pipe,X)

def makeTests(pipe,X):
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

    plotMarg(mu1_samples.T, filename=getPltDir()+"mu1_samples")
    plotMarg(mu2_samples.T, filename=getPltDir()+"mu2_samples")
    plotMarg(X.T, filename=getPltDir()+"data")
    
    # # --- 4 compare Posterior: ---
    compare_moments(samples,pipe,X)

def getPosteriorTarget(X, priorVar=20.0, cov = torch.eye(4)):
    N, d = X.shape
    Sigma0 = (priorVar ** 2) * torch.eye(d)
    Sigma0_inv = torch.linalg.inv(Sigma0)
    cov_inv = torch.linalg.inv(cov)

    Sigma_post = torch.linalg.inv(Sigma0_inv + N * cov_inv)
    mu_post = Sigma_post @ cov_inv @ X.sum(dim=0)

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
        cov=pipe.ModelParams["cov"],
    )
    # Differences
    mean_diff = mu_hat - muTrue
    cov_diff  = Sigma_hat - Sigma_true
    
    mean_l2 = torch.norm(mean_diff, p=2)       # Euclidean norm
    cov_fro = torch.norm(cov_diff, p='fro')    #

    print(f"mu_hat: {mu_hat}")
    print(f"Sigma_hat: {Sigma_hat}")
    print(f"muTrue: {muTrue}")
    print(f"sigmaTrue: {Sigma_true}")
    print(f"mean_diff: {mean_diff}")
    print(f"cov_diff: {cov_diff}")
    
    print(f"||mean_diff||_2 (Euclidean): {mean_l2}")
    print(f"||cov_diff||_F (Frobenius): {cov_fro}")

def scenario(config:NF_type, suffix = "_NSF_INDEP",isIndependentCopula:bool = False,FreezeWeights:bool = False):
    #standardize the data
    N             = 67108864
    known_cov     = getModelParams(isIndependentCopula)['cov']
    X             = sample_true_data(N, getTrueParams(),known_cov)
    nRestarts     = 1
    # standardize the data
    # mu         = X.mean(0)
    # std        = X.std(0)
    # X          = (X - mu) / std
    
    #config
    batch_size    = 1048576 #512 2048 128
    num_particles = 5
    
    #debug options
    runMarginals = False
    
    if runMarginals:
        pyro.clear_param_store()
        pipe_init_1  = pyroMarginalSVIPipeline(
            X[:,0:2],
            batch_size          = batch_size,
            indices             = slice(0, 2),
            num_particles       = num_particles,
            marginal_dim        = 2,
            plot_dir            = "toy_problem/plots/marginal_1" + suffix,
            config              = config,
            isIndependentCopula = isIndependentCopula
            )
        loss_list_1,_,flow_1_chkpt = pipe_init_1.trainModel(1000,10,1)
        plotLoss("svi_loss_diagnostic_init_1.png",loss_list_1)
        pipe_init_1.plot_samples(10000,"toy_problem/plots/MarginalSample_1.png")
        
        pyro.clear_param_store()
        pipe_init_2  = pyroMarginalSVIPipeline(
            X[:,2:4],
            batch_size          = batch_size,
            indices             = slice(2, 4),
            num_particles       = num_particles,
            marginal_dim        = 2,
            plot_dir            = "toy_problem/plots/marginal_2" + suffix,
            config              = config,
            isIndependentCopula = isIndependentCopula
            )
        loss_list_2,_,flow_2_chkpt = pipe_init_2.trainModel(1000,10,1)
        plotLoss("svi_loss_diagnostic_init_2.png",loss_list_2)
        pipe_init_2.plot_samples(10000,"toy_problem/plots/MarginalSample_2.png")
    else:
        flow_1_chkpt = None
        flow_2_chkpt = None
        
    tail_loss_list = []
    for run in range(nRestarts):
        ### --- Actual model ---
        pyro.clear_param_store()
        pipe = pyroCopulaSVIPipeline(
                X,
                batch_size          = batch_size,
                num_particles       = num_particles,
                flow_1_chkpt        = flow_1_chkpt,
                flow_2_chkpt        = flow_2_chkpt,
                plot_dir            = "toy_problem/plots/joint_SVI" + suffix,
                config              = config,
                isIndependentCopula = isIndependentCopula,
                FreezeWeights       = FreezeWeights
            )
        
        # --- pre-SVI unit tests for independent copula ---

        # test_independent_log_prob_matches_manual(pipe)
        # test_independent_sampling_matches_block_structure(pipe)
        # test_custom_q_against_known_block_gaussian(pipe)
        ## rebuild fresh pipeline for actual training  
        # pipe = pyroCopulaSVIPipeline(
        #         X,
        #         batch_size    = batch_size,
        #         num_particles = num_particles,
        #         flow_1_chkpt  = flow_1_chkpt,
        #         flow_2_chkpt  = flow_2_chkpt,
        #         plot_dir      = "toy_problem/joint_SVI"
        #     )
        
        ### calculates theoretical posterior (for our model) 
        ### and plots it (to be used for comparison)
        plot_posterior_marginals(pipe,X)  
        
        #--- main part of the model of the model ---
        Nsteps = 2000
        loss_list,diagnostics,_ = pipe.trainModel(Nsteps,100,run)
        tail_window = Nsteps//20
        tail_loss = np.median(loss_list[-tail_window:])
        tail_loss_list.append(tail_loss)
        best_score = np.inf 
        if tail_loss < best_score:
            best_score = tail_loss
            best_run = {
                "run": run,
                "tail_loss": tail_loss,
                "param_store": pyro.get_param_store().get_state(),
                "loss_list": loss_list,
            }
            postProcessTest(loss_list,diagnostics,pipe,X,run)
    print("best restart:", best_run["run"])
    print("best score:", best_run["tail_loss"])
    print("tail_loss_list:", tail_loss_list)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the toy problem scenario with CLI-configurable options."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=NF_type.NSF.value,
        choices=[member.value for member in NF_type],
        help="Flow type to use.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_NSF_DEBUG_6",
        help="Suffix appended to output directories.",
    )
    parser.add_argument(
        "--independent-copula",
        type=str2bool,
        default=False,
        help="Whether to use the independence copula (true/false).",
    )
    parser.add_argument(
        "--freeze-weights",
        type=str2bool,
        default=False,
        help="Whether to freeze pretrained marginal flow weights (true/false).",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    scenario(
        config=NF_type(args.config),
        suffix=args.suffix,
        isIndependentCopula=args.independent_copula,
        FreezeWeights=args.freeze_weights,
    )
    
if __name__ == "__main__":
    main()
    # ModelTest = pipe.make_q()
    # result = ModelTest.testInternalConsistency(100000,100000)
    print("end")
    #TODO: fix model -> sample from ML model