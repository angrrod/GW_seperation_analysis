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
import zuko.flows.spline as s
import numpy as np
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

    sample_1 = T_1(stdn64.icdf(Z_1).to(torch.float32)) #can be used because they are independent
    sample_2 = T_2(stdn64.icdf(Z_2).to(torch.float32))
    return torch.cat([sample_1,sample_2], dim=1).T

def logProbCopula(params,theta,d = 4):
    dtype=torch.float64
    stdn64 = torch.distributions.Normal(
        torch.tensor(0., dtype=dtype),
        torch.tensor(1., dtype=dtype),
    )
    paramsDerived   = buildDerivedParams(params)
    Sigma           = paramsDerived["Omega"].to(dtype=dtype)
    _, logabsdet    = torch.linalg.slogdet(Sigma)
    SigmaInv        = torch.linalg.inv(Sigma)
    PhiInv          = stdn64.icdf(theta)
    einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (SigmaInv - torch.eye(d).to(dtype=dtype)), PhiInv)  #sum in order to deal with quadratic form dimensions
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
    eps         = 1e-11 #numerical stability 
    stdn64 = torch.distributions.Normal( #-> cast inverse to get better precision
        torch.tensor(0., dtype=torch.float64),
        torch.tensor(1., dtype=torch.float64),
    )
    T_1,T_2,dist_1,dist_2  = exctractFlows(params)
    
    #get log prob of the flows using made products
    logp_marg_1 = dist_1.log_prob(theta[:,0:2])
    logp_marg_2 = dist_2.log_prob(theta[:,2:4])
    
    #generate input for log prob copula
    Q_1         = T_1.inv(theta[:,0:2])
    Q_2         = T_2.inv(theta[:,2:4])
    
    U_1         = stdn64.cdf(Q_1.double()).clamp(eps, 1.0 - eps)  #for numerical error
    U_2         = stdn64.cdf(Q_2.double()).clamp(eps, 1.0 - eps)
    U           = torch.concat([U_1,U_2],dim = 1)
    logDensity,logDetTerm,logCopulaTerm  = logProbCopula(params,U)
    total = logp_marg_1 + logp_marg_2 + logDensity
    return total,logp_marg_1,logp_marg_2,logDensity,logDetTerm,logCopulaTerm
def getTrueParams():
    ### custom 4d distribution, mixture of 3 guassians with different correlation structure -> multimodal and one can tune the connection
    weights = torch.tensor([0.4, 0.4,0.2])

    # Means (2 components × 4 dimensions)
    means = torch.tensor([
        [0., 0., 0., 0.],
        [2., 2., 2., 2.],
        [4., 0., 0., 4.]
    ])

    Sigma1 = torch.tensor([
        [1.0, 0.8, 0.2, 0.0],
        [0.8, 1.0, 0.3, 0.15],
        [0.2, 0.3, 1.0, 0.4],
        [0.0, 0.15, 0.4, 1.0]
    ])

    Sigma2 = torch.tensor([
        [1.0, 0.0, 0.4, 0.3],
        [0.0, 1.0, 0.1, 0.25],
        [0.4, 0.1, 1.0, 0.0],
        [0.3, 0.25, 0.0, 1.0]
    ])

    Sigma3 = torch.tensor([
        [1.0, -0.35, 0.2, 0.6],
        [-0.35, 1.0, -0.1, 0.15],
        [0.2, -0.1, 1.0, 0.0],
        [0.6, 0.15, 0.0, 1.0]
    ])

    # Covariances (3 × 4 × 4)
    covs = torch.stack([
        Sigma1,
        Sigma2,
        Sigma3
    ])

    params_true_data = {
            "covs" : covs,
            "means" : means,
            "weights" : weights
            }
    # components = MultivariateNormal(loc=params_true_data['means'], covariance_matrix=params_true_data['covs'])
    # mixture = MixtureSameFamily(mix, params_true_data['weights'])
    # samp_mix = mixture.sample((10000,))
    # plotMarg(samp_mix.T[0:2,:])
    return params_true_data

def get_params():
    params = {}
    # we start with a 4 d problem, get 2, 2d gaussians,
    # Use normalizing flows from U[0,1] distr to the gaussian, these are the quantiles
    d_1 = 2
    d_2 = 2
    flow_1 = zuko.flows.spline.NSF(
        features=d_1,  #dim of the output
        transforms=8,
    )
    flow_2 = zuko.flows.spline.NSF(
        features=d_2,  #dim of the output
        transforms=8,
    )
    params['flow_1'] = flow_1
    params['flow_2'] = flow_2
    Sigma_21 = torch.tensor([
            [0.2, 0.4],
            [0.8, 0.1]
        ])
    params["Sigma_21"] = pyro.param("Sigma_21", Sigma_21)
    params["Sigma_21_init"] = Sigma_21 
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

class VectorCopulaFlowQ(TorchDistribution):
    arg_constraints = {}  # fill if you have constrained params
    support = constraints.real_vector
    has_rsample = False  # set True if you implement rsample()
    def __init__(self,params):
        self.params = params
        batch_shape = torch.Size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([4]), validate_args=None)
        
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        total,_,_,_,_,_ = logProbVectorCopula(self.params,value)
        return total
    
    def sample(self, sample_shape=torch.Size()):
        N = math.prod(sample_shape) if len(sample_shape) > 0 else 1
        return sampleVectorCopulaModel(self.params,N).T #return (n,4)

#perform SVI optimization
class pyroCopulaSVIPipeline:
    def __init__(self,batch_size = 512):
        self.trueParams  = getTrueParams()
        self.modelParams = get_params()
        self.batch_size  = batch_size
        pyro.module("flow_1", self.modelParams['flow_1'])
        pyro.module("flow_2", self.modelParams['flow_2'])
        
    def model(self,X):
        pyro.module("flow_1", self.modelParams['flow_1'])
        pyro.module("flow_2", self.modelParams['flow_2'])
        
        Sigma_21 = pyro.param("Sigma_21", self.modelParams["Sigma_21_init"])
        
        params_now = dict(self.modelParams)
        params_now["Sigma_21"] = Sigma_21  #force the update
        q   = VectorCopulaFlowQ(params_now)
        N   = X.shape[0]
        with pyro.plate("data", size=N, subsample_size=self.batch_size) as ind:
            pyro.sample("x", q, obs=X[ind])

    def guide(self,X):
        return None

# # variational approx"
# # generate from copula
# # use normalizing flow to push forward
# svi = SVI(model, guide, Adam({"lr": 1e-2}), loss=Trace_ELBO())
def plotMarg(Z,filename,d = 4):
    X_np = Z.detach().cpu().numpy()
    plt.figure()
    plt.scatter(X_np[0,:], X_np[1,:], s=d)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    
# def max_param_delta(module, module_prev):
#     # module_prev is a list of cloned tensors
#     deltas = []
#     for p, p_prev in zip(module.parameters(), module_prev):
#         deltas.append((p.detach() - p_prev).abs().max().item())
#     return max(deltas) if deltas else 0.0

# def snapshot_params(module):
#     return [p.detach().clone() for p in module.parameters()]

# def summarize_params(module, *, include_grad=False):
#     rows = []
#     for name, p in module.named_parameters():
#         x = p.detach()
#         rows.append((
#             name,
#             tuple(x.shape),
#             x.min().item(),
#             torch.abs(x).min().item(),
#             x.max().item(),
#             x.mean().item(),
#             x.std(unbiased=False).item(),   # population stdev over entries
#         ))
#         if include_grad:
#             g = p.grad
#             if g is None:
#                 rows.append((name + " (grad)", tuple(x.shape), None, None, None, None))
#             else:
#                 gd = g.detach()
#                 rows.append((
#                     name + " (grad)",
#                     tuple(gd.shape),
#                     gd.min().item(),
#                     gd.max().item(),
#                     gd.mean().item(),
#                     gd.std(unbiased=False).item(),
#                 ))
#     return rows

# def print_summary(rows, header=""):
#     if header:
#         print(header)
#     print(f"{'param':60s} {'shape':14s} {'min':>12s} {'min(abs)':>12s} {'max':>12s} {'mean':>12s} {'std':>12s}")
#     for name, shape, mn, absMin, mx, mean, std in rows:
#         if mn is None:
#             print(f"{name:60s} {str(shape):14s} {'None':>12s} {'None':>12s} {'None':>12s} {'None':>12s} {'None':>12s}")
#         else:
#             print(f"{name:60s} {str(shape):14s} {mn:12.5g} {absMin:12.5g} {mx:12.5g} {mean:12.5g} {std:12.5g}")

def get_params_now(pipe):
    # pipe.modelParams contains references to flow modules etc.
    params_now = dict(pipe.modelParams)

    # overwrite anything learned via pyro.param
    params_now["Sigma_21"] = pyro.param("Sigma_21")

    # (optional) if you later move to other pyro.params, add them here too
    return params_now

if __name__ == "__main__":
    pyro.clear_param_store()
    N          = 8192
    batch_size = 512
    nSteps     = 200
    pipe       = pyroCopulaSVIPipeline(batch_size=batch_size)
    X          = sample_true_data(N, pipe.trueParams)
    mu = X.mean(0)
    std = X.std(0)

    X = (X - mu) / std
    #debug code
    # q = VectorCopulaFlowQ(get_params())
    # q.log_prob(X)
    
    svi        = SVI(pipe.model, pipe.guide, ClippedAdam({"lr": 1e-3, "clip_norm": 5.0}), loss=Trace_ELBO())
    step       = 0
    prev_loss  = 0
    diff       = 0
    loss_list = []
    diagnostics = []
    elbo = Trace_ELBO()
    while (step < nSteps):
        
        # prev_f1 = snapshot_params(pipe.modelParams["flow_1"])
        # prev_f2 = snapshot_params(pipe.modelParams["flow_2"])
        
        # loss as python float for logging
        loss = svi.step(X)
        # d1 = max_param_delta(pipe.modelParams["flow_1"], prev_f1)
        # d2 = max_param_delta(pipe.modelParams["flow_2"], prev_f2)
        # print("Δflow_1:", d1, "Δflow_2:", d2)
        
        diff = prev_loss - loss
        loss_list.append(loss)
        
        params_now = get_params_now(pipe)
        with torch.no_grad():
            total, m1, m2, cop,logDetTerm,logCopulaTerm = logProbVectorCopula(params_now, X)
            diag = dict(
                total=total.mean().item(),
                marg1=m1.mean().item(),
                marg2=m2.mean().item(),
                copula=cop.mean().item(),
                det=logDetTerm.mean().item(),
                gauss = logCopulaTerm.mean().item(),
            )
            diagnostics.append(diag)
        if step % 50 == 0:
            # for p in pyro.get_param_store().values():
            #     if p.grad is not None:
            #         p.grad.zero_()

            # differentiable scalar objective, no parameter update
            # dloss = elbo.differentiable_loss(pipe.model, pipe.guide, X)
            # pick one flow parameter tensor
            # p1 = next(pipe.modelParams["flow_1"].parameters())
            # Sigma = pyro.param("Sigma_21")
            # g1, g = torch.autograd.grad(
            #     dloss,
            #     (p1, Sigma),
            #     allow_unused=True
            # )
            # print("flow_1 grad is None?", g1 is None)
            # if g1 is not None:
            #     print("flow_1 grad max abs:", g1.abs().max().item())
            # # definitive dependency test (leaf/non-leaf safe)
            
            # print("autograd.grad(dloss, Sigma_21) is None?", g is None)
            # if g is not None:
            #     print("Sigma_21 grad max abs:", g.abs().max().item())
                
            PyroParams = {"Sigma_21": pyro.param("Sigma_21")}
            derived = buildDerivedParams(PyroParams)
            Omega = derived["Omega"]
            eigvals = torch.linalg.eigvalsh(Omega)
            print("")
            print(f"step: {step}, diff: {diff}")
            print(f"loss: {loss}, loss/N: {loss / N}")
            print(f"Omega: {Omega}")
            print(f"eigvals: {eigvals}")
            # rows_f1 = summarize_params(pipe.modelParams["flow_1"], include_grad=False)
            # rows_f2 = summarize_params(pipe.modelParams["flow_2"], include_grad=False)

            # print_summary(rows_f1, header="\n=== flow_1 parameter stats ===")
            # print_summary(rows_f2, header="\n=== flow_2 parameter stats ===")
        prev_loss = loss
        step +=1
        
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
    dist1 = params_now["flow_1"]()   # distribution on R^2
    T1 = dist1.transform
    dist2 = params_now["flow_2"]()   # distribution on R^2
    T2 = dist2.transform
    x_zuko_1 = dist1.sample((4096,))
    x_zuko_2 = dist2.sample((4096,))
    x_manual_1 = T1(x_zuko_1)
    x_manual_2 = T2(x_zuko_2)
    
    z1 = T1.inv(X[:,0:2])
    z2 = T2.inv(X[:,2:4])
    print(f"dist of T_1 data, mean: {x_manual_1.mean(0)}, var: {x_manual_1.var(0)}")
    print(f"dist of T_2 data, mean: {x_manual_2.mean(0)}, var: {x_manual_2.var(0)}")
    print(f"T_1 of data, mean: {z1.mean(0)}, var: {z1.var(0)}")
    print(f"T_2 of data, mean: {z2.mean(0)}, var: {z2.var(0)}")
    
    params_now = dict(pipe.modelParams)
    params_now["Sigma_21"] = pyro.param("Sigma_21")
    q = VectorCopulaFlowQ(params_now)
    samples = q.sample((5000,))
    plotMarg(samples.T[0:2,:],filename = "toy_problem/samplesT[0:2,:]")
    plotMarg(samples.T[2:4,:],filename = "toy_problem/samplesT[2:4,:]")
    plotMarg(X.T[0:2,:],filename = "toy_problem/X[0:2,:]")
    plotMarg(X.T[2:4,:],filename = "toy_problem/X[2:4,:]")

    
    print("end")