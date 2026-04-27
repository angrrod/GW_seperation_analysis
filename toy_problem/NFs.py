
from torch.distributions import Transform, constraints
import torch
from enum import Enum
import torch.nn as nn
from zuko.lazy import Flow, LazyTransform, UnconditionalDistribution
import torch.nn.functional as F

class NF_type(Enum):
    NSF    = "neural_spline_flow"
    MAF    = "masked_autoregressive_flow"
    AFFINE = "full_affine_flow"
    
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

    def _call(self, z: torch.Tensor) -> torch.Tensor:
        # Base -> data
        return z @ self.scale_tril.mT + self.loc

    def _inverse(self, x: torch.Tensor) -> torch.Tensor:
        # Data -> base
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

    def log_abs_det_jacobian(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        diag = torch.diagonal(self.scale_tril, dim1=-2, dim2=-1)
        ladj = torch.sum(torch.log(diag))
        return ladj.expand(z.shape[:-1])

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
