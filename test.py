from pathlib import Path

import numpy as np
import pandas as pd
import torch


omega_path = Path(
    "/root/phd/GW_seperation_analysis/"
    "postProcessing/Plots/copula_ml/"
    "fitted_copula_Omega.csv"
)


def test_cholesky(matrix: torch.Tensor, name: str) -> bool:
    """
    Attempt a Cholesky decomposition without raising immediately.
    """
    L, info = torch.linalg.cholesky_ex(
        matrix,
        check_errors=False,
    )

    info_value = int(info.item())
    success = info_value == 0

    print(f"\n{name}")
    print(f"  dtype:              {matrix.dtype}")
    print(f"  Cholesky succeeded: {success}")
    print(f"  cholesky_ex info:   {info_value}")

    if success:
        reconstruction_error = (
            L @ L.transpose(-1, -2) - matrix
        ).abs().max().item()

        print(
            "  reconstruction error:",
            f"{reconstruction_error:.6e}",
        )

    return success


# The CSV contains row names in its first column.
omega_df = pd.read_csv(
    omega_path,
    index_col=0,
)

omega_np = omega_df.to_numpy(dtype=np.float64)

if omega_np.ndim != 2:
    raise ValueError(
        f"Expected a 2D matrix, got shape {omega_np.shape}."
    )

if omega_np.shape[0] != omega_np.shape[1]:
    raise ValueError(
        f"Omega must be square, got shape {omega_np.shape}."
    )

if not np.isfinite(omega_np).all():
    raise ValueError("Omega contains NaN or Inf.")

omega64 = torch.from_numpy(omega_np)
omega32 = omega64.float()

print("=== LOADED OMEGA ===")
print("Shape:", tuple(omega64.shape))
print("Diagonal minimum:", torch.diagonal(omega64).min().item())
print("Diagonal maximum:", torch.diagonal(omega64).max().item())

# Numerical asymmetry introduced by triangular solves or CSV output.
asymmetry64 = (
    omega64 - omega64.transpose(-1, -2)
).abs().max().item()

asymmetry32 = (
    omega32 - omega32.transpose(-1, -2)
).abs().max().item()

print("Maximum asymmetry, float64:", f"{asymmetry64:.6e}")
print("Maximum asymmetry, float32:", f"{asymmetry32:.6e}")

# Symmetric versions for eigenvalue diagnostics.
omega64_sym = 0.5 * (
    omega64 + omega64.transpose(-1, -2)
)

omega32_sym = 0.5 * (
    omega32 + omega32.transpose(-1, -2)
)

eig64 = torch.linalg.eigvalsh(omega64_sym)
eig32 = torch.linalg.eigvalsh(omega32_sym)

print("\n=== EIGENVALUE DIAGNOSTICS ===")
print("float64 minimum eigenvalue:", f"{eig64.min().item():.12e}")
print("float64 maximum eigenvalue:", f"{eig64.max().item():.12e}")
print("float32 minimum eigenvalue:", f"{eig32.min().item():.12e}")
print("float32 maximum eigenvalue:", f"{eig32.max().item():.12e}")

if eig64.min() > 0:
    condition64 = (
        eig64.max() / eig64.min()
    ).item()

    print(
        "float64 spectral condition number:",
        f"{condition64:.12e}",
    )
else:
    print(
        "float64 spectral condition number: undefined "
        "(minimum eigenvalue is non-positive)"
    )

# Test all relevant combinations.
test_cholesky(
    omega64,
    "Raw CSV matrix in float64",
)

test_cholesky(
    omega64_sym,
    "Symmetrized matrix in float64",
)

test_cholesky(
    omega32,
    "Raw CSV matrix in float32",
)

test_cholesky(
    omega32_sym,
    "Symmetrized matrix in float32",
)

from torch.distributions import constraints

print("\n=== PYTORCH CONSTRAINT CHECK ===")

print(
    "Raw matrix symmetric under torch.isclose:",
    torch.isclose(
        omega32,
        omega32.T,
        rtol=1e-5,
        atol=1e-8,
    ).all().item(),
)

print(
    "Raw matrix PositiveDefinite constraint:",
    constraints.positive_definite.check(
        omega32
    ).item(),
)

print(
    "Symmetrized matrix PositiveDefinite constraint:",
    constraints.positive_definite.check(
        omega32_sym
    ).item(),
)