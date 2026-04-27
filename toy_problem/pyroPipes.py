import torch
import pyro
from abc import ABC,abstractmethod
from NFs import NF_type, FullAffineTransform,LazyFullAffineTransform
from pyro.infer import SVI, Trace_ELBO
from datetime import datetime
import os
from zuko.lazy import Flow, UnconditionalDistribution, UnconditionalTransform, LazyTransform
from zuko.distributions import DiagNormal
from zuko.transforms import IdentityTransform
import zuko
import pyro.distributions as dist
from VectorCopulaFlowQ import VectorCopulaFlowQ
from torch.distributions import constraints
import matplotlib.pyplot as plt
from pyro.contrib.zuko import ZukoToPyro
from setUp import getModelParams


def plot_all_2d_pairs_fixed(
    samples: torch.Tensor,
    filename: str,
    titles: list[str] | None = None,
    s: float = 5,
    alpha: float = 0.5,
    xlim: tuple[float, float] = (-5, 5),
    ylim: tuple[float, float] = (-5, 5),
    figsize: tuple[float, float] = (15, 10),
):
    """
    Plot all 6 pairwise 2D marginals of a 4D sample in a fixed 2x3 layout.
    """
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError(f"Expected samples of shape (N, 4), got {tuple(samples.shape)}")

    pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]

    if titles is None:
        titles = [f"Marginal [{i},{j}]" for i, j in pairs]

    if len(titles) != 6:
        raise ValueError(f"Expected exactly 6 titles, got {len(titles)}")

    samples_np = samples.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()

    for ax, (i, j), title in zip(axes, pairs, titles):
        ax.scatter(samples_np[:, i], samples_np[:, j], s=s, alpha=alpha)
        ax.set_xlabel(f"theta[{i}]")
        ax.set_ylabel(f"theta[{j}]")
        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
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

def plot_2d_samples(
    samples: torch.Tensor,
    filename: str,
    xlabel: str = "x0",
    ylabel: str = "x1",
    title: str | None = None,
    s: float = 5,
    alpha: float = 0.5,
):
    samples_np = samples.detach().cpu().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(samples_np[:, 0], samples_np[:, 1], s=s, alpha=alpha)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

class pyroPipe(ABC):
    def __init__(self,X,num_particles,batch_size,config = NF_type.NSF,isIndependentCopula:bool = False):
        self.ModelParams          = getModelParams(isIndependentCopula)
        self.batch_size           = batch_size
        self.optimizer            = pyro.optim.ClippedAdam(self._per_param_optim_args) 
        self.loss                 = Trace_ELBO(num_particles=num_particles) #pyro machinery
        self.X                    = X
        N,_                       = X.shape
        self.N                    = N
        self.has_q                = False
        self.plot_during_training = True
        self.config               = config
        self.isIndependentCopula  = isIndependentCopula
    
    @abstractmethod
    def model(self):
        raise NotImplementedError
    
    @abstractmethod
    def guide(self):
        raise NotImplementedError
    
    @abstractmethod
    def getOptimalState(self):
        raise NotImplementedError
    
    @abstractmethod
    def plot_training_state(self, step: int, n_samples: int = 5000):
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
                    total, m1, m2, cop, logDetTerm, logCopulaTerm = q.logProbVectorCopula(theta_sample)
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
                if self.plot_during_training:
                    self.plot_training_state(step=step, n_samples=5000)
                
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
        return {"lr": 1e-3, "clip_norm": 10.0}
    
    def plot_training_state(self, step: int, n_samples: int = 5000):
        self.flow.eval()
        with torch.no_grad():
            q = self.flow()
            samples = q.sample((n_samples,))

        filename = os.path.join(self.plot_dir, f"marginal_step_{step:06d}.png")

        plot_2d_samples(
            samples=samples,
            filename=filename,
            xlabel="theta[0]",
            ylabel="theta[1]",
            title=f"Marginal guide at step {step}",
        )

class pyroCopulaSVIPipeline(pyroPipe):
    def __init__(self,X,batch_size,num_particles, flow_1_chkpt = None, flow_2_chkpt = None,plot_dir = 'toy_problem',config = NF_type.NSF,isIndependentCopula = False,FreezeWeights:bool = False):
        super().__init__(X,num_particles,batch_size,config,isIndependentCopula)
        # Debug switches
        self.isIndependentCopula  = isIndependentCopula #do we use the independence copula
        self.useIdentityTransform = False #Marginal flows used is the special identity transform
        
        self.FreezeWeights = FreezeWeights
        self.has_q         = True
        self.plot_dir      = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

        if self.useIdentityTransform:
            self.flow_1 = make_identity_flow_module_2d()
            self.flow_2 = make_identity_flow_module_2d()
        else:
            if config == NF_type.MAF:
                self.flow_1 = zuko.flows.MAF(
                    features        = 2,
                    transforms      = 2,
                    hidden_features = (8,),
                )
                self.flow_2 = zuko.flows.MAF(
                    features        = 2,
                    transforms      = 2,
                    hidden_features = (8,),
                ) 
            elif config == NF_type.NSF:
                self.flow_1 = zuko.flows.NSF(
                    features        = 2,
                    transforms      = 2,
                    context         = 0,
                    hidden_features = (10,),
                    bins            = 8,
                    randperm        = False,
                )
                self.flow_2 = zuko.flows.NSF(
                    features        = 2,
                    transforms      = 2,
                    context         = 0,
                    hidden_features = (10,),
                    bins            = 8,
                    randperm        = False,
                )
            else: 
                self.flow_1 = make_full_affine_zuko_flow(features=2) 
                self.flow_2 = make_full_affine_zuko_flow(features=2) 
            
        self.B_init = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.5]
        ])
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
            covariance_matrix = self.ModelParams['cov']#(self.ModelParams["observedVar"] ** 2)*torch.eye(4)     # shape (2, 2, 2)
        ) 

        # tell pyro that all data is independent but samples the same parameter, the likelihood
        with pyro.plate("data", data.shape[0]):
            pyro.sample("obs", likl, obs=data) #gmm(theta)

    def guide(self, data):
        # pyro.module("flow", self.flow)
        q = self.make_q() #ZukoToPyro(self.flow())          # instantiate lazy flow -> actual distribution
        pyro.sample("theta", q)
        
    def make_q(self):
        #freeze? -> remove both lines from paramstore
        if not self.FreezeWeights:
            pyro.module("flow_1", self.flow_1)
            pyro.module("flow_2", self.flow_2)

        if self.isIndependentCopula:
            B    = None
            zeta = None
        else:
            B_raw      = pyro.param("B_raw", self.B_init) #B = to existing parameter in param store or if it is not present, self.B_init
            zeta_raw   = pyro.param("zeta_raw", self.zeta_init, constraint=constraints.positive)
            B          = 0.1 * torch.tanh(B_raw)
            zeta       = zeta_raw
            
        q = VectorCopulaFlowQ(
            flow_1               = self.flow_1,
            flow_2               = self.flow_2,
            B                    = B,
            zeta                 = zeta,
            isIndependentCopula  = self.isIndependentCopula,
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

    def plot_training_state(self, step: int, n_samples: int = 5000):
        with torch.no_grad():
            q = self.make_q()
            samples = q.sample((n_samples,))   # (n_samples, 4)

        filename = os.path.join(self.plot_dir, f"joint_pairs_step_{step:06d}.png")

        plot_all_2d_pairs_fixed(
            samples=samples,
            filename=filename,
            titles=[
                f"[0,1] step {step}",
                f"[0,2] step {step}",
                f"[0,3] step {step}",
                f"[1,2] step {step}",
                f"[1,3] step {step}",
                f"[2,3] step {step}",
            ],
            xlim=(-5, 5),
            ylim=(-5, 5),
            figsize=(15, 10),
        )        

class pyroMarginalSVIPipeline(pyroPipe): 
    def __init__(self,X,num_particles,indices,batch_size = 512,marginal_dim = 2,plot_dir = "toy_problem",config = NF_type.NSF,isIndependentCopula = False):
        super().__init__(X,num_particles,batch_size,config,isIndependentCopula)
        self.marginal_dim = marginal_dim
        self.plot_dir     = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        if config == NF_type.MAF:
            self.flow = zuko.flows.MAF(
                features        = 2,
                transforms      = 2,
                hidden_features = (8,),
            )
        elif config == NF_type.NSF:
            self.flow = zuko.flows.NSF(
                    features        = 2,
                    transforms      = 2,
                    context         = 0,
                    hidden_features = (10,),
                    bins            = 8,
                    randperm        = False,
            )
        else:
            self.flow         = make_full_affine_zuko_flow(features=self.marginal_dim)
        
        self.ModelParams  = getModelParams(isIndependentCopula)
        self.indices      = indices
        
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
            covariance_matrix = self.ModelParams["cov"][self.indices,self.indices] #self.ModelParams['cov'][self.indices,self.indices] #    # shape (2, 2, 2)
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

    def plot_samples(self, N: int, filename: str = "MarginalSample.png"):
        """
        Draw N samples from the trained guide (NSF flow)
        and create a 2D scatter plot.

        Parameters
        ----------
        N : int
            Number of samples
        filename : str | None
            If provided, saves the plot to this path
        """
        import matplotlib.pyplot as plt

        self.flow.eval()  # important: disable any training-time behavior

        with torch.no_grad():
            # instantiate the trained flow distribution
            dist = self.flow()  # Zuko distribution
            samples = dist.sample((N,))  # shape (N, 2)

        samples_np = samples.detach().cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(samples_np[:, 0], samples_np[:, 1], s=5, alpha=0.5)
        plt.xlabel("theta[0]")
        plt.ylabel("theta[1]")
        plt.title(f"{N} samples from trained NSF guide")

        plt.savefig(filename, dpi=300, bbox_inches="tight")

        plt.close()
