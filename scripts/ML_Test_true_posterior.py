
import matplotlib.pyplot as plt
import torch
import numpy as np
import corner
from dingo.core.posterior_models.vector_copula_Model import CopulaNormalizingFlowModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
from setUpLoggerScenario import setUpLoggerScenario
from pathlib import Path
import os
import pandas as pd
from matplotlib.lines import Line2D
from collections.abc import Sequence
import wandb
import time

################################ ML ##############################

COPULA_PARAMETER_NAMES = {"B", "z"}

def is_copula_parameter(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in COPULA_PARAMETER_NAMES

def get_loss_components(model, theta):
    """
    Return marginal, copula and full joint NLL.
    """
    distribution = model.network.distribution()

    components = distribution.log_prob_components(theta)

    marginal_nll = -components[
        "logp_marg_total"
    ].mean()

    copula_nll = -components[
        "log_copula_total"
    ].mean()

    joint_nll = -components[
        "log_prob_total"
    ].mean()

    return {
        "marginal": marginal_nll,
        "copula": copula_nll,
        "joint": joint_nll,
    }

def configure_training_phase(network, phase: str):
    """
    phase:
        "marginal" : train marginal flows only
        "copula"   : train B,z only
        "joint"    : train everything
    """

    if phase not in {"marginal", "copula", "joint"}:
        raise ValueError(f"Unknown training phase: {phase}")

    network.train()

    marginal_parameters = []
    copula_parameters = []

    for name, parameter in network.named_parameters():
        if is_copula_parameter(name):
            copula_parameters.append(parameter)
        else:
            marginal_parameters.append(parameter)

    # ---------------------------------------------------------
    # Select trainable parameters.
    # ---------------------------------------------------------
    train_marginals = phase in {"marginal", "joint"}
    train_copula = phase in {"copula", "joint"}

    for parameter in marginal_parameters:
        parameter.requires_grad_(train_marginals)

    for parameter in copula_parameters:
        parameter.requires_grad_(train_copula)

    # ---------------------------------------------------------
    # IMPORTANT:
    # frozen marginal flows must also have frozen BatchNorm
    # running statistics.
    # ---------------------------------------------------------
    if phase == "copula":
        for flow in network.flows:
            flow.eval()
    else:
        for flow in network.flows:
            flow.train()

    return marginal_parameters, copula_parameters

def create_phase_optimizer(
    network,
    phase,
    n_epochs,
    lr,
    lr_min,
):
    """
    Create a fresh optimizer/scheduler for one alternating phase.
    """

    marginal_parameters, copula_parameters = (
        configure_training_phase(
            network,
            phase,
        )
    )

    if phase == "marginal":
        parameters = marginal_parameters

    elif phase == "copula":
        parameters = copula_parameters

    else:
        parameters = (
            marginal_parameters
            + copula_parameters
        )

    parameters = [
        p for p in parameters
        if p.requires_grad
    ]

    if not parameters:
        raise RuntimeError(
            f"No trainable parameters in phase {phase}"
        )

    optimizer = torch.optim.Adam(
        parameters,
        lr=lr,
    )

    gamma = (
        lr_min / lr
    ) ** (1.0 / n_epochs)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=gamma,
    )

    return optimizer, scheduler, parameters

def _atomic_torch_save(payload: dict, output_path):
    """
    Write a checkpoint atomically.

    The temporary file is replaced only after torch.save succeeds, reducing
    the chance of leaving a corrupt checkpoint after an interruption.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)

def save_copula_ml_checkpoint(
    output_path,
    *,
    model,
    metadata,
    parameter_names,
    loss_history,
    completed_epochs,
    best_loss,
    optimizer=None,
    scheduler=None,
    data_source=None,
):
    checkpoint = {
        "checkpoint_format_version": 1,
        "completed_epochs": int(completed_epochs),
        "best_loss": float(best_loss),
        "loss_history": [
            float(loss) for loss in loss_history
        ],

        # The actual fitted model.
        "model_state_dict": model.network.state_dict(),

        # Required to reconstruct exactly the same architecture.
        "metadata": metadata,

        # Required to preserve the HDF5 column ordering.
        "parameter_names": [
            str(name) for name in parameter_names
        ],

        "data_source": (
            str(data_source)
            if data_source is not None
            else None
        ),

        # These are None for a weights-only checkpoint.
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),

        # Useful for approximately continuing shuffled minibatches.
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }

    _atomic_torch_save(
        checkpoint,
        output_path,
    )

    print(f"Saved checkpoint: {output_path}")

def _optimizer_to_device(optimizer, device):
    """
    Move tensors stored inside an optimizer state to the target device.
    """
    device = torch.device(device)

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)

def load_copula_ml_checkpoint(
    checkpoint_path,
    *,
    device="cpu",
    restore_optimizer=True,
):
    checkpoint_path = Path(checkpoint_path)

    # Only load checkpoints you created yourself.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    metadata = checkpoint["metadata"]

    model = CopulaNormalizingFlowModel(
        metadata=metadata,
        device=device,
    )

    model.network.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    optimizer = None
    scheduler = None

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
    )
    scheduler_state = checkpoint.get(
        "scheduler_state_dict"
    )

    if restore_optimizer and optimizer_state is not None:
        saved_lr = optimizer_state["param_groups"][0]["lr"]

        optimizer = torch.optim.Adam(
            model.network.parameters(),
            lr=saved_lr,
        )

        optimizer.load_state_dict(
            optimizer_state
        )

        _optimizer_to_device(
            optimizer,
            device,
        )

        if scheduler_state is not None:
            saved_gamma = scheduler_state.get(
                "gamma",
                1.0,
            )

            scheduler = (
                torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=saved_gamma,
                )
            )

            scheduler.load_state_dict(
                scheduler_state
            )

    rng_state = checkpoint.get("torch_rng_state")

    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu())

    cuda_rng_state = checkpoint.get(
        "cuda_rng_state_all"
    )

    if (
        cuda_rng_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_rng_state
        )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(
        "Completed epochs:",
        checkpoint.get("completed_epochs", 0),
    )
    print(
        "Best stored loss:",
        checkpoint.get("best_loss"),
    )

    return model, optimizer, scheduler, checkpoint

def plot_ml_losses(
    losses,
    output_path,
    moving_average_window=10,
    skip_initial_epochs=20,
):
    losses = np.asarray(losses, dtype=float)

    if losses.ndim != 1 or losses.size == 0:
        raise ValueError(
            f"Expected a non-empty one-dimensional loss array, "
            f"got shape {losses.shape}."
        )

    if not np.isfinite(losses).all():
        raise ValueError("Loss history contains NaN or Inf.")

    epochs = np.arange(1, losses.size + 1)

    skip = int(skip_initial_epochs)
    if skip < 0 or skip >= losses.size:
        raise ValueError(
            f"skip_initial_epochs must be between 0 and "
            f"{losses.size - 1}, got {skip}."
        )

    # Apply the same trimming to every plotted diagnostic.
    plotted_losses = losses[skip:]
    plotted_epochs = epochs[skip:]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(
        plotted_epochs,
        plotted_losses,
        linewidth=1.0,
        alpha=0.45,
        label="Training NLL",
    )

    window = min(
        int(moving_average_window),
        plotted_losses.size,
    )

    if window >= 2:
        kernel = np.ones(window, dtype=float) / window

        smoothed = np.convolve(
            plotted_losses,
            kernel,
            mode="valid",
        )

        # The first moving-average value corresponds to the final epoch
        # in the first averaging window.
        smoothed_epochs = plotted_epochs[window - 1:]

        ax.plot(
            smoothed_epochs,
            smoothed,
            linewidth=2.0,
            label=f"{window}-epoch moving average",
        )

    # Minimum among the epochs actually displayed.
    best_index = int(np.argmin(plotted_losses))
    best_epoch = int(plotted_epochs[best_index])
    best_loss = float(plotted_losses[best_index])

    ax.scatter(
        best_epoch,
        best_loss,
        marker="o",
        label=f"Minimum: {best_loss:.6g} (epoch {best_epoch})",
        zorder=3,
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Negative log-likelihood per sample")
    ax.set_title(
        "Unconditional copula maximum-likelihood training"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

def print_fitted_copula_model(
    model,
    parameter_names,
    output_dir=None,
):
    """
    Print and optionally save the fitted non-amortized copula parameters.

    B and raw z are the optimized parameterization. Omega is the implied
    normalized copula matrix and is the more interpretable fitted object.
    """
    network = model.network

    if getattr(network, "conditional", True):
        raise ValueError(
            "print_fitted_copula_model() expects an unconditional copula model."
        )

    if not hasattr(network, "B") or not hasattr(network, "z"):
        raise AttributeError(
            "The unconditional network must expose trainable B and z parameters."
        )

    parameter_names = list(parameter_names)
    if len(parameter_names) != network.D:
        raise ValueError(
            f"Received {len(parameter_names)} parameter names, but model D={network.D}."
        )

    network.eval()
    with torch.no_grad():
        fitted_distribution = network.distribution()

        B = network.B.detach().cpu().squeeze(0)
        z_raw = network.z.detach().cpu().reshape(-1)
        zeta = torch.nn.functional.softplus(z_raw)
        omega = fitted_distribution.Omega().detach().cpu().squeeze(0)

    factor_names = [f"factor_{j + 1}" for j in range(B.shape[1])]
    B_df = pd.DataFrame(
        B.numpy(),
        index=parameter_names,
        columns=factor_names,
    )
    omega_df = pd.DataFrame(
        omega.numpy(),
        index=parameter_names,
        columns=parameter_names,
    )

    block_dims = list(network.block_dims)
    if sum(block_dims) != network.D:
        raise ValueError(
            f"block_dims={block_dims} do not sum to D={network.D}."
        )

    if len(block_dims) == 2:
        split = block_dims[0]
        omega_cross_df = omega_df.iloc[:split, split:]
    else:
        omega_cross_df = None

    print("\n=== FINAL FITTED NON-AMORTIZED COPULA MODEL ===")
    print(f"Event dimension D: {network.D}")
    print(f"Low-rank dimension P: {B.shape[1]}")
    print(f"Block dimensions: {block_dims}")
    print(f"Raw z: {z_raw.numpy()}")
    print(f"zeta = softplus(z): {zeta.numpy()}")

    print("\nB factor matrix:")
    print(B_df.to_string(float_format=lambda x: f"{x: .6f}"))

    print("\nImplied normalized copula matrix Omega:")
    print(omega_df.to_string(float_format=lambda x: f"{x: .6f}"))

    if omega_cross_df is not None:
        print("\nCross-block copula matrix Omega_AB:")
        print(
            omega_cross_df.to_string(
                float_format=lambda x: f"{x: .6f}"
            )
        )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        B_path = output_dir / "fitted_copula_B.csv"
        omega_path = output_dir / "fitted_copula_Omega.csv"
        B_df.to_csv(B_path)
        omega_df.to_csv(omega_path)

        print("\nSaved fitted copula parameters:")
        print(f"B:     {B_path}")
        print(f"Omega: {omega_path}")

        if omega_cross_df is not None:
            omega_cross_path = output_dir / "fitted_copula_Omega_AB.csv"
            omega_cross_df.to_csv(omega_cross_path)
            print(f"Omega_AB: {omega_cross_path}")

    return {
        "B": B,
        "z_raw": z_raw,
        "zeta": zeta,
        "Omega": omega,
        "B_dataframe": B_df,
        "Omega_dataframe": omega_df,
        "Omega_AB_dataframe": omega_cross_df,
    }

def load_best_model(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.network.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.network.to(device)

    return model, checkpoint

def train_copula_ML(
    model,
    train_loader,

    phases,
    n_epochs,
    lr,
    lr_min,

    device="cuda",
    checkpoint_path="postProcessing/copula_ml/copula_ml_best_independent.pt",
    start_from_best=False,
    wandb_run=None,
    # Diagnostics after every stage
    diagnostic_training_data=None,
    diagnostic_parameter_names=None,
    diagnostic_selected_parameters=None,
    diagnostic_output_dir=None,
    diagnostic_n_model_samples=5000,
    diagnostic_max_training_samples=5000,
):
    
    # =========================================================
    # VALIDATE TRAINING SCHEDULE
    # =========================================================

    phases = list(phases)
    n_epochs = list(n_epochs)
    lr = list(lr)
    lr_min = list(lr_min)

    n_stages = len(phases)

    if n_stages == 0:
        raise ValueError(
            "Training schedule must contain at least one stage."
        )

    if not (
        len(n_epochs)
        == len(lr)
        == len(lr_min)
        == n_stages
    ):
        raise ValueError(
            "phases, n_epochs, lr and lr_min must have "
            "the same length.\n"
            f"Got:\n"
            f"  phases:   {len(phases)}\n"
            f"  n_epochs: {len(n_epochs)}\n"
            f"  lr:       {len(lr)}\n"
            f"  lr_min:   {len(lr_min)}"
        )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    device = torch.device(device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested, but CUDA is unavailable."
        )
    previous_epochs = 0
    best_loss = float("inf")

    # =========================================================
    # LOAD EXISTING BEST MODEL
    # =========================================================
    if start_from_best:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Cannot continue training because no checkpoint exists at:\n"
                f"{checkpoint_path}"
            )

        model,checkpoint = load_best_model(model,checkpoint_path,device)
        stored_lrs = checkpoint.get("learning_rates")
            
        best_loss = float(checkpoint["best_loss"])
        previous_epochs = int(
            checkpoint.get("epoch", 0)
        )

        print("\nLoaded best copula model")
        print(f"Checkpoint:       {checkpoint_path}")
        print(f"Stored epoch:     {previous_epochs}")
        print(f"Stored best NLL:  {best_loss:.6f}")
        if stored_lrs is not None:
            print("Stored learning rates:")
            for group_name, group_lr in stored_lrs.items():
                print(f"  {group_name}: {group_lr:.3e}")
        else:
            print("Stored learning rates: unavailable in older checkpoint")
    else:
        print("\nStarting copula training from current initialization.")
    model.network.to(device)
    losses = []

    # Continue W&B epoch numbering after the loaded checkpoint.
    absolute_epoch = previous_epochs

    if wandb_run is not None:
        wandb_run.config.update(
            {
                "device": str(device),

                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),

                "training_phases": phases,
                "training_epochs": n_epochs,
                "training_lr": lr,
                "training_lr_min": lr_min,
                "n_training_stages": n_stages,

                "start_from_best": bool(
                    start_from_best
                ),

                "checkpoint_path": str(
                    checkpoint_path
                ),
            },
            allow_val_change=True,
        )


    # =========================================================
    # TRAIN EACH STAGE
    # =========================================================
    for stage_index in range(n_stages):

        phase = phases[stage_index]

        n_phase_epochs = int(
            n_epochs[stage_index]
        )

        phase_lr = float(
            lr[stage_index]
        )

        phase_lr_min = float(
            lr_min[stage_index]
        )

        print("\n" + "=" * 70)

        print(
            f"TRAINING STAGE "
            f"{stage_index + 1}/{n_stages}"
        )

        print(
            f"Phase:  {phase.upper()}"
        )

        print(
            f"Epochs: {n_phase_epochs}"
        )

        print(
            f"LR:     {phase_lr:.3e} "
            f"-> {phase_lr_min:.3e}"
        )

        print("=" * 70)

        (
            optimizer,
            scheduler,
            active_parameters,
        ) = create_phase_optimizer(
            network=model.network,
            phase=phase,
            n_epochs=n_phase_epochs,
            lr=phase_lr,
            lr_min=phase_lr_min,
        )
        # =====================================================
        # EPOCH LOOP FOR THIS PHASE
        # =====================================================
        for local_epoch in tqdm(
            range(n_phase_epochs)
        ):

            if device.type == "cuda":

                torch.cuda.reset_peak_memory_stats(
                    device
                )

                torch.cuda.synchronize(
                    device
                )

            epoch_start_time = (
                time.perf_counter()
            )

            epoch_objective = 0.0
            epoch_marginal = 0.0
            epoch_copula = 0.0
            epoch_joint = 0.0

            n_seen = 0

            # =================================================
            # MINI-BATCH LOOP
            # =================================================
            for theta in train_loader:

                theta = theta.to(
                    device=device,
                    non_blocking=(
                        device.type == "cuda"
                    ),
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                # ---------------------------------------------
                # Compute all three objectives.
                # ---------------------------------------------
                loss_components = (
                    get_loss_components(
                        model,
                        theta,
                    )
                )

                # This is THE objective actually optimized
                # during the current phase.
                loss = loss_components[phase]

                if not torch.isfinite(loss):
                    raise ValueError(
                        "Non-finite loss at "
                        f"epoch {absolute_epoch}, "
                        f"phase={phase}: {loss}"
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    active_parameters,
                    max_norm=10.0,
                )

                optimizer.step()

                batch_size = theta.shape[0]

                epoch_objective += (
                    loss.detach().cpu().item()
                    * batch_size
                )

                epoch_marginal += (
                    loss_components[
                        "marginal"
                    ]
                    .detach()
                    .cpu()
                    .item()
                    * batch_size
                )

                epoch_copula += (
                    loss_components[
                        "copula"
                    ]
                    .detach()
                    .cpu()
                    .item()
                    * batch_size
                )

                epoch_joint += (
                    loss_components[
                        "joint"
                    ]
                    .detach()
                    .cpu()
                    .item()
                    * batch_size
                )

                n_seen += batch_size

            # =================================================
            # EPOCH SUMMARY
            # =================================================
            average_objective = (
                epoch_objective / n_seen
            )

            average_marginal = (
                epoch_marginal / n_seen
            )

            average_copula = (
                epoch_copula / n_seen
            )

            average_joint = (
                epoch_joint / n_seen
            )

            absolute_epoch += 1

            losses.append(
                average_joint
            )

            # -------------------------------------------------
            # VERY IMPORTANT:
            #
            # Do NOT compare marginal loss during one phase
            # against copula loss during another.
            #
            # The common checkpoint criterion is always the
            # complete JOINT NLL.
            # -------------------------------------------------
            if average_joint < best_loss:

                best_loss = average_joint

                current_lr = float(
                    optimizer.param_groups[0]["lr"]
                )

                torch.save(
                    {
                        "model_state_dict":
                            model.network.state_dict(),

                        "best_loss":
                            best_loss,

                        "epoch":
                            absolute_epoch,

                        "phase":
                            phase,

                        "stage":
                            stage_index + 1,

                        "learning_rates": {
                            phase: current_lr
                        },
                    },
                    checkpoint_path,
                )

                print(
                    "\nSaved new best model: "
                    f"epoch={absolute_epoch}, "
                    f"stage={stage_index + 1}, "
                    f"phase={phase}, "
                    f"joint NLL={best_loss:.6f}"
                )

            current_lr = float(
                optimizer.param_groups[0]["lr"]
            )

            if device.type == "cuda":

                torch.cuda.synchronize(
                    device
                )

            epoch_seconds = (
                time.perf_counter()
                - epoch_start_time
            )

            # =================================================
            # W&B
            # =================================================
            if wandb_run is not None:

                metrics = {
                    "epoch":
                        absolute_epoch,

                    "train/objective_nll":
                        float(
                            average_objective
                        ),

                    "train/marginal_nll":
                        float(
                            average_marginal
                        ),

                    "train/copula_nll":
                        float(
                            average_copula
                        ),

                    "train/joint_nll":
                        float(
                            average_joint
                        ),

                    "train/best_joint_nll":
                        float(best_loss),

                    "train/epoch_seconds":
                        float(
                            epoch_seconds
                        ),

                    "train/samples_seen":
                        int(n_seen),

                    "learning_rate/current":
                        current_lr,

                    "training/stage":
                        int(stage_index + 1),

                    "training/phase_epoch":
                        int(local_epoch + 1),

                    # W&B accepts strings here.
                    "training/phase":
                        phase,
                }

                if device.type == "cuda":

                    metrics.update(
                        {
                            (
                                "gpu/"
                                "peak_allocated_gib"
                            ): (
                                torch.cuda
                                .max_memory_allocated(
                                    device
                                )
                                / 2**30
                            ),

                            (
                                "gpu/"
                                "peak_reserved_gib"
                            ): (
                                torch.cuda
                                .max_memory_reserved(
                                    device
                                )
                                / 2**30
                            ),
                        }
                    )

                wandb_run.log(
                    metrics,
                )

            scheduler.step()
            
        # =====================================================
        # END-OF-STAGE SAMPLE DIAGNOSTIC
        # =====================================================
        if (
            diagnostic_training_data is not None
            and diagnostic_parameter_names is not None
            and diagnostic_output_dir is not None
        ):
            diagnostic_output_dir = Path(
                diagnostic_output_dir
            )

            diagnostic_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            stage_number = stage_index + 1

            diagnostic_path = (
                diagnostic_output_dir
                / (
                    f"stage_{stage_number:02d}_"
                    f"{phase}_"
                    f"epoch_{absolute_epoch:06d}.png"
                )
            )

            print(
                "\nGenerating end-of-stage diagnostic:"
            )
            print(
                f"  stage: {stage_number}/{n_stages}"
            )
            print(
                f"  phase: {phase}"
            )
            print(
                f"  epoch: {absolute_epoch}"
            )
            print(
                f"  output: {diagnostic_path}"
            )

            # -------------------------------------------------
            # IMPORTANT:
            # This samples the CURRENT model at the end of this
            # stage, not the globally best checkpoint.
            # -------------------------------------------------
            plot_ml_training_vs_samples_corner(
                model=model,
                training_data=diagnostic_training_data,
                parameter_names=(
                    diagnostic_parameter_names
                ),
                selected_parameters=(
                    diagnostic_selected_parameters
                ),
                output_path=diagnostic_path,
                n_model_samples=(
                    diagnostic_n_model_samples
                ),
                max_training_samples=(
                    diagnostic_max_training_samples
                ),
                bins=20,
                dpi = 100,
                quantile_range=(0.005, 0.995),
                random_seed=1234,
                plot_title=(
                    f"Stage {stage_number}/{n_stages} | "
                    f"{phase} | epoch {absolute_epoch}"
                ),            
            )

            # -------------------------------------------------
            # Upload to W&B.
            #
            # Use the SAME W&B key each time. This means W&B
            # stores the diagnostic as a time series indexed by
            # absolute_epoch, which is exactly what we want.
            # -------------------------------------------------
            if wandb_run is not None:
                try:
                    wandb_run.log(
                        {
                            "diagnostics/stage_corner": (
                                wandb.Image(
                                    str(diagnostic_path),
                                    caption=(
                                        f"Stage {stage_number}/{n_stages}: "
                                        f"{phase}, epoch {absolute_epoch}"
                                    ),
                                )
                            ),
                            "diagnostics/stage": stage_number,
                            "diagnostics/phase": phase,
                        },
                    )

                except Exception as exc:
                    print(
                        "\nWARNING: could not upload "
                        "stage diagnostic to W&B:"
                    )
                    print(exc)

            print(
                "End-of-stage diagnostic completed."
            )
    # =========================================================
    # RESTORE GLOBAL BEST MODEL
    # =========================================================
    best_checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.network.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.network.eval()

    # Restore requires_grad as well, so the model isn't left
    # partially frozen after training.
    for parameter in (
        model.network.parameters()
    ):
        parameter.requires_grad_(True)

    print(
        "\nRestored best model "
        "after alternating training"
    )

    print(
        f"Best epoch: "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best joint NLL: "
        f"{best_checkpoint['best_loss']:.6f}"
    )

    print(
        f"Best phase: "
        f"{best_checkpoint.get('phase', 'unknown')}"
    )

    if wandb_run is not None:

        wandb_run.summary[
            "best_epoch"
        ] = int(
            best_checkpoint["epoch"]
        )

        wandb_run.summary[
            "best_nll"
        ] = float(
            best_checkpoint["best_loss"]
        )

        wandb_run.summary[
            "best_phase"
        ] = best_checkpoint.get(
            "phase",
            "unknown",
        )

        wandb_run.summary[
            "checkpoint_path"
        ] = str(
            checkpoint_path
        )

    return losses

def make_copula_ml_metadata(theta_dim,block_dims,p,is_independent=False):
    return {
        "dataset_settings": {
            "type": "posterior_samples_ml",
            "theta_dim": theta_dim,
        },
        "train_settings": {
            "model": {
                "posterior_model_type": "copula_normalizing_flow",
                "posterior_kwargs": {
                    "conditional": False,
                    "block_dims": block_dims,
                    "flows": {
                        "flow_1": {
                            "num_flow_steps": 6,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 32,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                        "flow_2": {
                            "num_flow_steps": 6,
                            "base_transform_kwargs": {
                                "hidden_dim": 64,
                                "num_transform_blocks": 4,
                                "activation": "elu",
                                "dropout_probability": 0.0,
                                "batch_norm": True,
                                "num_bins": 32,
                                "base_transform_type": "rq-coupling",
                            },
                        },
                    },
                    "CopulaKwargs": {
                        "P": p,
                        "is_independent" : is_independent
                    },
                },

                # Explicitly absent.
                "embedding_kwargs": None,
            }
        },
    }

def _samples_to_2d_numpy(samples, name: str) -> np.ndarray:
    """
    Convert training or generated samples to shape [N, D].

    Supported inputs:
        [N, D]
        [N, 1, D]
        [1, N, D]
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    else:
        samples = np.asarray(samples)

    if samples.ndim == 3:
        if samples.shape[1] == 1:
            # Unconditional copula output: [N, 1, D]
            samples = samples[:, 0, :]
        elif samples.shape[0] == 1:
            # Alternative Dingo convention: [1, N, D]
            samples = samples[0, :, :]
        else:
            raise ValueError(
                f"{name} has ambiguous three-dimensional shape "
                f"{samples.shape}. Expected [N, 1, D] or [1, N, D]."
            )

    if samples.ndim != 2:
        raise ValueError(
            f"{name} must reduce to shape [N, D], "
            f"but has shape {samples.shape}."
        )

    if not np.isfinite(samples).all():
        raise ValueError(f"{name} contains NaN or Inf.")

    return samples

def plot_ml_training_vs_samples_corner(
    model,
    training_data,
    parameter_names,
    output_path,
    *,
    selected_parameters=None,
    n_model_samples=5000,
    max_training_samples=5000,
    bins=20,
    dpi = 100,
    quantile_range=(0.005, 0.995),
    random_seed=1234,
    plot_title=None,
    data_mean=None,
    data_std=None,
):
    """
    Overlay training data and samples from a fitted unconditional ML model.

    Parameters
    ----------
    model
        Trained CopulaNormalizingFlowModel.

    training_data
        Original tensor used for training, usually `theta`, with shape [N, D].

    parameter_names
        Names corresponding to the D columns, for example `result.columns`.

    output_path
        Output PNG path.

    selected_parameters
        Optional list of parameter names to include. If None, all dimensions
        are plotted.

    n_model_samples
        Number of samples drawn from the fitted ML model.

    max_training_samples
        Maximum number of original training samples shown. Subsampling only
        affects the plot, not the trained model.

    quantile_range
        Quantile limits used to determine common plotting ranges. This avoids
        a few extreme samples distorting every panel.

    Returns
    -------
    model_samples : np.ndarray
        All generated model samples with shape [n_model_samples, D].
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parameter_names = [str(name) for name in parameter_names]

    training_np = _samples_to_2d_numpy(
        training_data,
        name="training_data",
    )

    if training_np.shape[1] != len(parameter_names):
        raise ValueError(
            f"training_data has D={training_np.shape[1]}, but "
            f"{len(parameter_names)} parameter names were supplied."
        )
        
    # Batch-normalization layers must use their fitted running statistics.
    model.network.eval()

    with torch.no_grad():
        generated = model.sample(
            num_samples=n_model_samples,
        )

    model_np = _samples_to_2d_numpy(
        generated,
        name="model samples",
    )
    
    # ---------------------------------------------------------
    # Transform standardized coordinates back to physical units.
    # ---------------------------------------------------------
    if data_mean is not None or data_std is not None:

        if data_mean is None or data_std is None:
            raise ValueError(
                "data_mean and data_std must either both be supplied "
                "or both be None."
            )

        if torch.is_tensor(data_mean):
            data_mean = data_mean.detach().cpu().numpy()
        else:
            data_mean = np.asarray(data_mean)

        if torch.is_tensor(data_std):
            data_std = data_std.detach().cpu().numpy()
        else:
            data_std = np.asarray(data_std)

        data_mean = data_mean.reshape(-1)
        data_std = data_std.reshape(-1)

        if data_mean.shape[0] != training_np.shape[1]:
            raise ValueError(
                f"data_mean has D={data_mean.shape[0]}, but "
                f"training data have D={training_np.shape[1]}."
            )

        if data_std.shape[0] != training_np.shape[1]:
            raise ValueError(
                f"data_std has D={data_std.shape[0]}, but "
                f"training data have D={training_np.shape[1]}."
            )

        # z -> x = z * sigma + mu
        training_np = (
            training_np * data_std[None, :]
            + data_mean[None, :]
        )

        model_np = (
            model_np * data_std[None, :]
            + data_mean[None, :]
        )
        

    if model_np.shape[1] != training_np.shape[1]:
        raise ValueError(
            f"Model samples have D={model_np.shape[1]}, while the "
            f"training data have D={training_np.shape[1]}."
        )

    # Select columns by parameter name.
    if selected_parameters is None:
        selected_indices = np.arange(training_np.shape[1])
        selected_names = parameter_names
    else:
        selected_parameters = [
            str(name) for name in selected_parameters
        ]

        missing = [
            name
            for name in selected_parameters
            if name not in parameter_names
        ]

        if missing:
            raise KeyError(
                f"Requested corner-plot parameters were not found: {missing}"
            )

        selected_indices = np.array(
            [parameter_names.index(name) for name in selected_parameters],
            dtype=int,
        )
        selected_names = selected_parameters

    if len(selected_names) > 12:
        print(
            f"Warning: plotting {len(selected_names)} dimensions creates "
            f"{len(selected_names) ** 2} corner panels."
        )

    training_selected = training_np[:, selected_indices]
    model_selected = model_np[:, selected_indices]

    # Subsample the training cloud to prevent it from dominating the figure.
    rng = np.random.default_rng(random_seed)

    n_training_plot = min(
        int(max_training_samples),
        training_selected.shape[0],
    )

    if n_training_plot < training_selected.shape[0]:
        selected_rows = rng.choice(
            training_selected.shape[0],
            size=n_training_plot,
            replace=False,
        )
        training_plot = training_selected[selected_rows]
    else:
        training_plot = training_selected

    # Use common robust ranges for both data clouds.
    combined = np.vstack(
        [training_plot, model_selected]
    )

    lower_q, upper_q = quantile_range

    if not 0.0 <= lower_q < upper_q <= 1.0:
        raise ValueError(
            "quantile_range must satisfy "
            "0 <= lower < upper <= 1."
        )

    lower = np.quantile(combined, lower_q, axis=0)
    upper = np.quantile(combined, upper_q, axis=0)

    span = upper - lower

    # Add a small margin and handle nearly constant parameters.
    padding = np.where(
        span > 0,
        0.05 * span,
        0.01 * np.maximum(np.abs(lower), 1.0),
    )

    plotting_ranges = [
        (float(lo - pad), float(hi + pad))
        for lo, hi, pad in zip(lower, upper, padding)
    ]

    # Draw the original training distribution first.
    fig = corner.corner(
        training_plot,
        labels=selected_names,
        range=plotting_ranges,
        bins=bins,
        color="C0",
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=False,
        smooth=1.0,
        smooth1d=1.0,
        hist_kwargs={
            "alpha": 0.45,
        },
        contour_kwargs={
            "linewidths": 1.2,
        },
        label_kwargs={
            "fontsize": 9,
        },
        show_titles=True,
        title_fmt=".4g",
        title_kwargs={
            "fontsize": 9,
        },
    )

    # Overlay samples from the fitted ML model.
    corner.corner(
        model_selected,
        labels=selected_names,
        range=plotting_ranges,
        bins=bins,
        color="C1",
        fig=fig,
        plot_datapoints=False,
        plot_density=True,
        plot_contours=True,
        fill_contours=False,
        smooth=1.0,
        smooth1d=1.0,
        hist_kwargs={
            "alpha": 0.45,
        },
        contour_kwargs={
            "linewidths": 1.2,
        },
        show_titles=False,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color="C0",
            linewidth=2.0,
            label=f"Training data (n={training_plot.shape[0]})",
        ),
        Line2D(
            [],
            [],
            color="C1",
            linewidth=2.0,
            label=f"Fitted ML model (n={model_selected.shape[0]})",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
    )
    if plot_title is None:
        plot_title = (
            "Training posterior samples "
            "versus fitted copula model"
        )

    fig.suptitle(
        plot_title,
        y=1.01,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved corner plot to: {output_path}")
    print(f"Training samples plotted: {training_plot.shape[0]}")
    print(f"Model samples plotted:    {model_selected.shape[0]}")
    print(f"Parameters plotted:       {selected_names}")

    return model_np

def create_copula_optimizer_and_scheduler(
    network: torch.nn.Module,
    n_epochs: int,
    lr: float = 2e-3,
    lr_min: float = 2e-5,
    copula_lr_factor: float = 1e-2,
    copula_parameter_names: Sequence[str] = ("B", "z"),
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    """
    Create an Adam optimizer with separate learning rates for:

    1. Copula parameters, identified by their final parameter name.
    2. All remaining trainable parameters.

    Both learning-rate groups decay exponentially to the same absolute
    minimum learning rate after `n_epochs` scheduler steps.

    Parameters
    ----------
    network
        Neural network containing the trainable parameters.
    n_epochs
        Number of training epochs and scheduler steps.
    lr
        Initial learning rate for non-copula parameters.
    lr_min
        Final learning rate for both parameter groups.
    copula_lr_factor
        Multiplicative factor applied to `lr` for copula parameters.
        A value of 1e-2 makes their initial learning rate two orders
        of magnitude smaller.
    copula_parameter_names
        Final parameter names used to identify copula parameters.
        For example, "copula.B" and "module.copula.B" both match "B".

    Returns
    -------
    optimizer
        Adam optimizer with separate parameter groups.
    scheduler
        Group-specific exponential learning-rate scheduler.
    """
    if n_epochs <= 0:
        raise ValueError("n_epochs must be positive.")

    if lr <= 0.0 or lr_min <= 0.0:
        raise ValueError("Learning rates must be positive.")

    copula_lr = lr * copula_lr_factor

    if lr < lr_min:
        raise ValueError(
            f"Non-copula initial LR ({lr:.3e}) is smaller than "
            f"lr_min ({lr_min:.3e})."
        )

    if copula_lr < lr_min:
        raise ValueError(
            f"Copula initial LR ({copula_lr:.3e}) is smaller than "
            f"lr_min ({lr_min:.3e})."
        )

    copula_name_set = set(copula_parameter_names)

    copula_parameters = []
    non_copula_parameters = []

    copula_names_found = []
    non_copula_names_found = []

    for full_name, parameter in network.named_parameters():
        if not parameter.requires_grad:
            continue

        leaf_name = full_name.rsplit(".", maxsplit=1)[-1]

        if leaf_name in copula_name_set:
            copula_parameters.append(parameter)
            copula_names_found.append(full_name)
        else:
            non_copula_parameters.append(parameter)
            non_copula_names_found.append(full_name)

    if not copula_parameters:
        available_parameters = [
            name
            for name, parameter in network.named_parameters()
            if parameter.requires_grad
        ]

        raise ValueError(
            "No copula parameters were found.\n"
            f"Requested names: {sorted(copula_name_set)}\n"
            f"Available trainable parameters: {available_parameters}"
        )

    if not non_copula_parameters:
        raise ValueError(
            "No non-copula trainable parameters were found."
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params": non_copula_parameters,
                "lr": lr,
                "group_name": "non_copula",
            },
            {
                "params": copula_parameters,
                "lr": copula_lr,
                "group_name": "copula",
            },
        ]
    )

    # These decay factors differ because the groups start at different
    # learning rates but must reach the same absolute lr_min.
    non_copula_gamma = (lr_min / lr) ** (1.0 / n_epochs)
    copula_gamma = (lr_min / copula_lr) ** (1.0 / n_epochs)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            lambda epoch, gamma=non_copula_gamma: gamma**epoch,
            lambda epoch, gamma=copula_gamma: gamma**epoch,
        ],
    )

    print("\nOptimizer parameter groups")
    print(
        f"Non-copula: {lr:.3e} -> {lr_min:.3e}"
    )
    # for name in non_copula_names_found:
    #     print(f"  {name}")

    print(
        f"Copula:     {copula_lr:.3e} -> {lr_min:.3e}"
    )
    # for name in copula_names_found:
    #     print(f"  {name}")

    return optimizer, scheduler

def require_cuda_device(device_index: int = 0) -> torch.device:
    """Return a CUDA device, failing rather than silently using the CPU."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Check that the job has a GPU and that the installed PyTorch build "
            "has CUDA support."
        )

    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)

    print(f"Using CUDA device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    return device

def initialize_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity=None,
    name=None,
    config=None,
):
    """Initialize W&B only when experiment tracking is enabled."""
    if not enabled:
        return None

    if wandb is None:
        raise ImportError(
            "Weights & Biases logging is enabled, but wandb is not installed. "
            "Install it with `pip install wandb` and authenticate with "
            "`wandb login` or WANDB_API_KEY."
        )

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config or {},
    )

def initialize_dependent_from_independent(
    model,
    independent_checkpoint_path,
    device,
):
    checkpoint = torch.load(
        independent_checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    source_state = checkpoint["model_state_dict"]
    target_state = model.network.state_dict()

    for name, value in source_state.items():

        # Do NOT copy the independent copula parameters.
        # Keep fresh B,z from the dependent model.
        if is_copula_parameter(name):
            continue

        if name not in target_state:
            raise KeyError(
                f"{name} from independent checkpoint "
                "not found in dependent model."
            )

        if target_state[name].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {name}: "
                f"{value.shape} vs "
                f"{target_state[name].shape}"
            )

        target_state[name] = value

    model.network.load_state_dict(
        target_state,
        strict=True,
    )

    return checkpoint

#TODO: add validation + early stopping
def main_hdf5():
    q0_path = Path(
        "postProcessing/copula_ml/"
        "copula_ml_best_independent.pt"
    )

    q1_path = Path(
        "postProcessing/copula_ml/"
        "copula_ml_best_copula_only.pt"
    )

    q2_path = Path(
        "postProcessing/copula_ml/"
        "copula_ml_best_joint.pt"
    )
    # phases = [
    #     "marginal",
    #     "copula",
    #     "marginal",
    #     "copula",
    #     "marginal",
    #     "copula",
    #     "marginal",
    #     "copula",
    #     "marginal",
    #     "copula",
    #     "marginal",
    #     "copula",
    #     # "marginal",
    #     # "copula",
    #     # "marginal",
    #     # "copula",
    #     "joint",
    # ]

    # n_epochs = [
    #     2000,
    #     1000,
    #     2000,
    #     1000,
    #     2000,
    #     1000,
    #     2000,
    #     1000,
    #     5000,
    #     2000,
    #     5000,
    #     2000,
    #     # 5000,
    #     # 2000,
    #     # 5000,
    #     # 4000,
    #     5000,
    # ]

    # lr = [
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     2e-3,
    #     5e-4,
    #     5e-4,
    #     1e-4,
    #     1e-4,
    #     # 1e-5,
    #     # 1e-5,
    #     # 2e-6,
    #     # 2e-6,
    #     1e-4,
    # ]

    # lr_min = [
    #     5e-5,
    #     5e-5,
    #     5e-5,
    #     5e-5,
    #     2e-5,
    #     2e-5,
    #     2e-5,
    #     2e-5,
    #     1e-4,
    #     1e-4,
    #     1e-5,
    #     1e-5,
    #     # 2e-6,
    #     # 2e-6,
    #     # 1e-6,
    #     # 1e-6,
    #     5e-7,
    # ]
    phases   = ['copula']
    n_epochs = [5000]
    lr       = [1e-3]
    lr_min   = [1e-6]
    start_from_best = True

    # ---------------------------------------------------------
    # Change 11: select CUDA and fail if no GPU is available.
    # ---------------------------------------------------------
    device = require_cuda_device(
        device_index=0
    )

    # ---------------------------------------------------------
    # Change 12: central training and W&B configuration.
    # ---------------------------------------------------------
    use_wandb = True
    wandb_project = "copula-ml"

    # Set this to your W&B entity/team when necessary.
    # Leaving it as None uses your default W&B entity.
    wandb_entity = None

    # None lets W&B generate a run name.
    wandb_run_name = None

    batch_size = 16384

    # Build scenario.

    (
        scenario,
        logger,
        plot_dir,
        data_dir,
    ) = setUpLoggerScenario()

    data_dir = (
        data_dir / "results_joint_final.hdf5"
    )

    results = utils.exctractResults(
        data_dir,
        logger,
    )

    result = utils.join_waveform_posteriors(
        results
    )

    # preferred_parameters = [
    #     # =========================
    #     # Signal A: 14 dimensions
    #     # =========================
    #     "chirp_mass_A",
    #     "mass_ratio_A",
    #     "a_1_A",
    #     "a_2_A",
    #     "tilt_1_A",
    #     "tilt_2_A",
    #     "phi_12_A",
    #     "phi_jl_A",
    #     "theta_jn_A",
    #     "psi_A",
    #     "phase_A",
    #     "geocent_time_A",
    #     "ra_A",
    #     "dec_A",

    #     # =========================
    #     # Signal B: 14 dimensions
    #     # =========================
    #     "chirp_mass_B",
    #     "mass_ratio_B",
    #     "a_1_B",
    #     "a_2_B",
    #     "tilt_1_B",
    #     "tilt_2_B",
    #     "phi_12_B",
    #     "phi_jl_B",
    #     "theta_jn_B",
    #     "psi_B",
    #     "phase_B",
    #     "geocent_time_B",
    #     "ra_B",
    #     "dec_B",
    # ]
    preferred_parameters = result.columns

    corner_parameters = [
        name
        for name in preferred_parameters
        if name in result.columns
    ]

    if len(corner_parameters) < 2:
        corner_parameters = list(
            result.columns[:6]
            )
    diagnostics_dir = (
        Path(plot_dir)
        / "copula_ml"
    )

    stage_diagnostics_dir = (
        diagnostics_dir
        / "training_stages"
    )

    stage_diagnostics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    # ---------------------------------------------------------
    # Change 13: replace the original DataLoader.
    #
    # The full tensor remains on the CPU. pin_memory=True allows
    # the later non-blocking CPU-to-CUDA batch transfers.
    # ---------------------------------------------------------
    
    result_ml = result[preferred_parameters].copy()
    theta_raw = torch.from_numpy(
        result_ml.to_numpy(dtype=np.float32)
    )
    
    theta_mean = theta_raw.mean(dim=0)
    theta_std = theta_raw.std(dim=0)

    # Safety against an accidentally constant parameter
    theta_std = torch.clamp(theta_std, min=1e-8)

    theta = (
        theta_raw - theta_mean
    ) / theta_std
    
    train_loader = DataLoader(
        theta,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    # ---------------------------------------------------------
    # Change 14: construct the model on CUDA.
    # ---------------------------------------------------------
    metadata = make_copula_ml_metadata( 
        theta_dim=30,
        block_dims=[15, 15],
        p=16,
        is_independent=False,
    )

    model = CopulaNormalizingFlowModel(
        metadata=metadata,
        device=device.type,
    )

    # Defensive explicit transfer of all registered parameters
    # and buffers.
    model.network.to(device)
    q0_checkpoint = initialize_dependent_from_independent(
        model=model,
        independent_checkpoint_path=q0_path,
        device=device,
    )
    # ---------------------------------------------------------
    # Change 15: initialize W&B after data and model settings
    # have been resolved.
    # ---------------------------------------------------------
    run = initialize_wandb_run(
        enabled=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        config={
            "data_source": str(data_dir),
            "n_training_samples": int(
                theta.shape[0]
            ),
            "theta_dim": int(theta.shape[1]),
            "batch_size": int(batch_size),
            "training_phases": phases,
            "training_epochs": n_epochs,
            "training_lr": lr,
            "training_lr_min": lr_min,
            "n_training_stages": len(phases),
            "start_from_best": bool(
                start_from_best
            ),
            "model_metadata": metadata,
        },
    )

    # ---------------------------------------------------------
    # Change 16: wrap training and diagnostics in try/finally,
    # and pass both device and run into train_copula_ML().
    # ---------------------------------------------------------
    try:
        losses_copula  = train_copula_ML(
            model=model,
            train_loader=train_loader,


            phases=["copula"],
            n_epochs=[5000],
            lr=[1e-3],
            lr_min=[1e-6],

            device=device,
            checkpoint_path=q1_path,
            start_from_best=False,
            wandb_run=run,

            diagnostic_training_data=theta,

            diagnostic_parameter_names=(
                result_ml.columns
            ),

            diagnostic_selected_parameters=(
                corner_parameters
            ),

            diagnostic_output_dir=(
                stage_diagnostics_dir
            ),

            diagnostic_n_model_samples=2000,
            diagnostic_max_training_samples=2000,
        )


        losses_joint = train_copula_ML(
            model=model,
            train_loader=train_loader,

            phases=["joint"],
            n_epochs=[20000],
            lr=[1e-4],
            lr_min=[1e-6],

            device=device,

            checkpoint_path=q2_path,
            start_from_best=False,

            wandb_run=run,

            diagnostic_training_data=theta,
            diagnostic_parameter_names=result_ml.columns,
            diagnostic_selected_parameters=corner_parameters,
            diagnostic_output_dir=stage_diagnostics_dir,

            diagnostic_n_model_samples=2000,
            diagnostic_max_training_samples=2000,
        )
        # -----------------------------------------------------
        # Change 17: give diagnostic files named paths, because
        # those same paths are used when uploading to W&B.
        # -----------------------------------------------------
        diagnostics_dir = (
            Path(plot_dir) / "copula_ml"
        )

        corner_plot_path = (
            diagnostics_dir
            / "training_vs_ml_corner.png"
        )

        # This original optional diagnostic can remain here.
        # print_fitted_copula_model(
        #     model,
        #     parameter_names=result.columns,
        #     output_dir=diagnostics_dir,
        # )


        generated_samples = (
            plot_ml_training_vs_samples_corner(
                model=model,
                training_data=theta,
                parameter_names=result_ml.columns,
                selected_parameters=(
                    corner_parameters
                ),
                output_path=corner_plot_path,
                n_model_samples=1000,
                max_training_samples=1000,
                bins=20,
                dpi=100,
                quantile_range=(0.005, 0.995),
                random_seed=1234,
                plot_title=None,
                data_mean=theta_mean,
                data_std=theta_std,
            )
        )

        # -----------------------------------------------------
        # Change 18: upload the generated diagnostic images.
        # Add this after both plotting functions have completed.
        # -----------------------------------------------------
        if run is not None:
            run.log(
                {
                    (
                        "diagnostics/"
                        "training_vs_ml_corner"
                    ): wandb.Image(
                        str(corner_plot_path)
                    ),
                }
            )

        print("end")

    # ---------------------------------------------------------
    # Change 19: always close W&B, including when training or
    # plotting raises an exception.
    # ---------------------------------------------------------
    finally:
        if run is not None:
            run.finish()

if __name__ == "__main__":
    # import os
    # import torch
    # available_cpus = len(os.sched_getaffinity(0))

    # # Leave one or two cores for Python and operating-system work.
    # n_threads = min(14, available_cpus)

    # torch.set_num_threads(n_threads)
    # torch.set_num_interop_threads(1)

    # print("Available CPUs:      ", available_cpus)
    # print("PyTorch threads:     ", torch.get_num_threads())
    # print("Interop threads:     ", torch.get_num_interop_threads())
        
    main_hdf5()