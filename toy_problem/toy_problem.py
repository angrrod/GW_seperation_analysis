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
import pyro.distributions.constraints as pyro_constraints
import math
import inspect
import zuko.flows.spline as s

def sampleGaussianBase(params,N):
    paramsDerived   = buildDerivedParams(params)
    dist = torch.distributions.MultivariateNormal(loc=torch.tensor([0.,0.,0.,0.])
                                                , covariance_matrix=paramsDerived["Omega"])
    sample = dist.sample((N,))
    return sample

def sampleVectorCopula(params,N):
    eps  = 1e-6
    res  = sampleGaussianBase(params,N)  # replace with multivariate distr
    dist = torch.distributions.Normal(loc=0, scale=1) #coincidence
    Z    = dist.cdf(res).clamp(eps, 1.0 - eps)
    return Z[:,0:2],Z[:,2:4]

def sampleVectorCopulaModel(params,N):
    Z_1,Z_2 = sampleVectorCopula(params,N)
    dist    = torch.distributions.Normal(loc=0, scale=1)
    T_1,T_2,_,_  = exctractFlows(params)

    sample_1 = T_1(dist.icdf(Z_1)) #can be used because they are independent
    sample_2 = T_2(dist.icdf(Z_2))
    return torch.cat([sample_1,sample_2], dim=1).T

def logProbCopula(params,theta,d = 4):
    dist            = torch.distributions.Normal(loc=0, scale=1)
    paramsDerived   = buildDerivedParams(params)
    Sigma           = paramsDerived["Omega"]
    _, logabsdet    = torch.linalg.slogdet(Sigma)
    SigmaInv        = torch.linalg.inv(Sigma)
    PhiInv          = dist.icdf(theta)
    einsum          = torch.einsum("ni,ij,nj->n", PhiInv, (SigmaInv - torch.eye(d)), PhiInv)  #sum in order to deal with quadratic form dimensions
    logDensity      = (-1/2)*logabsdet+(-1/2)*einsum
    return logDensity

def exctractFlows(params):
    dist_1 = params['flow_1']()
    T_1    = dist_1.transform
    dist_2 = params['flow_2']()
    T_2    = dist_2.transform
    return T_1,T_2,dist_1,dist_2

def logProbVectorCopula(params,theta):
    eps     = 1e-6 #numerical stability
    dist    = torch.distributions.Normal(loc=0, scale=1)
    T_1,T_2,dist_1,dist_2  = exctractFlows(params)
    Q_1         = T_1.inv(theta[:,0:2])
    Q_2         = T_2.inv(theta[:,2:4])
    U_1         = dist.cdf(Q_1).clamp(eps, 1.0 - eps)  #for numerical error
    U_2         = dist.cdf(Q_2).clamp(eps, 1.0 - eps)
    logp_marg_1 = dist_1.log_prob(Q_1)
    logp_marg_2 = dist_2.log_prob(Q_2)
    U           = torch.concat([U_1,U_2],dim = 1)
    logDensity  = logProbCopula(params,U)
    return logp_marg_1 + logp_marg_2 + logDensity
    
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
    
    return params

def ScaleSigma(Sigma_21):
    smax = torch.linalg.svdvals(Sigma_21).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    return Sigma_21 * (1.0 - 1e-4) / (scale + 1e-4)

def buildDerivedParams(params_orig):
    # build derived matrix
    params = {}  
    Sigma_21 = ScaleSigma(params_orig['Sigma_21'])
    smax = torch.linalg.svdvals(Sigma_21).max()
    scale = torch.maximum(
        smax,
        torch.tensor(1.0)
    )
    Sigma_21 = Sigma_21 * (1.0 - 1e-4) / (scale + 1e-4) #add numerical stability element
    top             = torch.cat([torch.eye(2), Sigma_21.T], dim=1)
    bottom          = torch.cat([Sigma_21, torch.eye(2)], dim=1)
    Omega           = torch.cat([top, bottom], dim=0)
    params["Omega"] = Omega
    params["SigmaScaled"] = Sigma_21
    #matrix will be pos definite if shur compliment I - A^T*A is positive definite which translates to |v|^2 > |Av|^2 for all v and is the case if the largest eigenvalue is smaller than 1 or the spectral norm is smaller than 1
    return params

def sample_true_data(N:int,true_params):
    z = Categorical(probs=true_params['weights']).sample((N,))          # (N,)

    # gather component parameters per datapoint
    mu = true_params['means'][z]                                       # (N, d)
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
        return logProbVectorCopula(self.params,value)
    
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
        q   = VectorCopulaFlowQ(self.modelParams)
        N   = X.shape[0]
        with pyro.plate("data", size=N, subsample_size=self.batch_size) as ind:
            pyro.sample("x", q, obs=X[ind])

    def guide(self,X):
        pass

# # variational approx"
# # generate from copula
# # use normalizing flow to push forward
# svi = SVI(model, guide, Adam({"lr": 1e-2}), loss=Trace_ELBO())
def plotMarg(Z,d = 4):
    X_np = Z.detach().cpu().numpy()
    plt.figure()
    plt.scatter(X_np[0,:], X_np[1,:], s=d)
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()
    
if __name__ == "__main__":
    pyro.clear_param_store()
    N          = 4096
    batch_size = 512
    nSteps     = 10000
    pipe       = pyroCopulaSVIPipeline(batch_size=batch_size)
    X          = sample_true_data(N, pipe.trueParams)
    svi        = SVI(pipe.model, pipe.guide, ClippedAdam({"lr": 1e-4, "clip_norm": 5.0}), loss=Trace_ELBO())
    step       = 0
    prev_loss  = 0
    diff       = 0
    diff_list = []
    while (step < nSteps) and (abs(diff) > 1e-3 or step == 0):
        # loss as python float for logging
        loss = svi.step(X)
        diff = prev_loss - loss
        diff_list.append(diff)
        
        if step % 100 == 0:
            print("Sigma_21 grad max abs:",
                pyro.param("Sigma_21").grad.abs().max().item()
                if pyro.param("Sigma_21").grad is not None else None)
            PyroParams = {"Sigma_21&": pyro.param("Sigma_21")}
            derived = buildDerivedParams(PyroParams)
            Omega = derived["Omega"]
            eigmin = torch.linalg.eigvalsh(Omega).min().item()
            smax = torch.linalg.svdvals(ScaleSigma(PyroParams["Sigma_21"])).max().item()
            max_param = pyro.param("Sigma_21").abs().max().item()

            with torch.no_grad():
                f1_l2 = torch.sqrt(sum((p**2).sum() for p in pipe.modelParams["flow_1"].parameters())).item()
                f2_l2 = torch.sqrt(sum((p**2).sum() for p in pipe.modelParams["flow_2"].parameters())).item()

            print(f"step: {step}, diff: {diff}")
            print(f"loss: {loss}, loss/N: {loss / N}")
            print(f"max|Sigma_21|: {max_param}")
            print(f"smax(A): {smax}, eigmin(Omega): {eigmin}")
            print(f"f1_norm: {f1_l2}, f2_norm: {f2_l2}\n")
            
        prev_loss = loss
        step +=1
        
    # ---- plot diagnostic curve ----
    plt.figure()
    plt.plot(diff_list)
    plt.xlabel("step")
    plt.ylabel("prev_loss - loss")
    plt.title("SVI loss difference diagnostic")
    plt.grid(True)
    filename = "svi_loss_diagnostic.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    q = VectorCopulaFlowQ(pipe.modelParams)
    samples = q.sample((1000,))
    print(samples.shape)
    # plotMarg
    print("end")