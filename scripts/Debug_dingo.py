from Pipeline import DINGO_pipeline
from scenario.setUpLoggerScenario import setUpLoggerScenario
import corner
import matplotlib.pyplot as plt
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.base_model import BasePosteriorModel
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

###########################
###   Run Debug Code    ###
###########################

TRAIN                 = True
INFER                 = True
PLOT                  = True
DIAGNOSE_IO           = False
NUM_SAMPLES_TO_PLOT   = 20000

def diagnose_dingo_io(
    DP,
    batch_size=64,
    repeats=2,
):
    """
    Diagnose Dingo input performance without modifying Dingo.

    Measures:
        1. HDF5 layout
        2. contiguous reads without transforms
        3. scattered reads without transforms
        4. scattered reads with Dingo transforms
    """

    import time
    import h5py
    import numpy as np

    print("\n========================================")
    print("DINGO I/O DIAGNOSTIC")
    print("========================================")

    print("local_settings:")
    print(DP.local_settings)

    # This will also perform the local-file copy if
    # local_cache_path has been configured.
    DP.prepare_training()

    wfd = DP.waveform_dataset

    print("\nWaveformDataset")
    print("file:", wfd.file_name)
    print("length:", len(wfd))
    print("transform:", type(wfd.transform))

    # -------------------------------------------------
    # Inspect HDF5 structure/chunking
    # -------------------------------------------------

    print("\n=== HDF5 LAYOUT ===")

    with h5py.File(wfd.file_name, "r") as f:

        def inspect(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"{name}\n"
                    f"    shape       = {obj.shape}\n"
                    f"    dtype       = {obj.dtype}\n"
                    f"    chunks      = {obj.chunks}\n"
                    f"    compression = {obj.compression}"
                )

        f.visititems(inspect)

    # -------------------------------------------------
    # Dataset-access timings
    # -------------------------------------------------

    rng = np.random.default_rng(12345)

    original_transform = wfd.transform

    contiguous_times = []
    scattered_raw_times = []
    scattered_transform_times = []

    def close_handle():
        if wfd.file_handle is not None:
            wfd.file_handle.close()
            wfd.file_handle = None

    try:

        for r in range(repeats):

            # Use different portions of the file for
            # different repetitions.
            start = (
                r * 10 * batch_size
            ) % (
                len(wfd) - batch_size
            )

            contiguous_idx = list(
                range(
                    start,
                    start + batch_size,
                )
            )

            # Sorted so the request is still spatially
            # scattered but friendly to h5py fancy indexing.
            scattered_idx = np.sort(
                rng.choice(
                    len(wfd),
                    size=batch_size,
                    replace=False,
                )
            ).tolist()

            # ---------------------------
            # Contiguous, no transforms
            # ---------------------------

            close_handle()
            wfd.transform = None

            t0 = time.perf_counter()

            data = wfd.__getitems__(
                contiguous_idx
            )

            dt = time.perf_counter() - t0

            contiguous_times.append(dt)

            del data

            print(
                f"repeat {r}: "
                f"contiguous/raw = {dt:.3f} s"
            )

            # ---------------------------
            # Scattered, no transforms
            # ---------------------------

            close_handle()
            wfd.transform = None

            t0 = time.perf_counter()

            data = wfd.__getitems__(
                scattered_idx
            )

            dt = time.perf_counter() - t0

            scattered_raw_times.append(dt)

            del data

            print(
                f"repeat {r}: "
                f"scattered/raw  = {dt:.3f} s"
            )

            # ---------------------------
            # Scattered + transformations
            # ---------------------------

            close_handle()
            wfd.transform = original_transform

            t0 = time.perf_counter()

            data = wfd.__getitems__(
                scattered_idx
            )

            dt = time.perf_counter() - t0

            scattered_transform_times.append(
                dt
            )

            del data

            print(
                f"repeat {r}: "
                f"scattered/full = {dt:.3f} s"
            )

    finally:

        wfd.transform = original_transform
        close_handle()

    print("\n=== SUMMARY ===")

    print(
        "contiguous / raw     : "
        f"{np.mean(contiguous_times):.3f} s"
    )

    print(
        "scattered / raw      : "
        f"{np.mean(scattered_raw_times):.3f} s"
    )

    print(
        "scattered / transforms: "
        f"{np.mean(scattered_transform_times):.3f} s"
    )

    print("========================================\n")

def chirp_q_to_component_masses(chirp_mass, mass_ratio):
    """
    Convert chirp mass and Bilby's q = m2 / m1 into ordered
    component masses. Supports scalars and NumPy arrays.
    """
    chirp_mass = np.asarray(chirp_mass, dtype=float)
    q = np.asarray(mass_ratio, dtype=float)

    chirp_mass, q = np.broadcast_arrays(chirp_mass, q)

    if np.any(chirp_mass <= 0.0):
        raise ValueError("chirp_mass must be strictly positive.")

    if np.any((q <= 0.0) | (q > 1.0)):
        raise ValueError(
            "Expected Bilby mass_ratio q=m2/m1 in (0, 1]."
        )

    mass_1 = (
        chirp_mass
        * (1.0 + q) ** (1.0 / 5.0)
        / q ** (3.0 / 5.0)
    )
    mass_2 = q * mass_1

    if mass_1.ndim == 0:
        return float(mass_1), float(mass_2)

    return mass_1, mass_2

def make_corner_plot(samples, truth_parameters, out_file,DP):
    """
    Make a corner plot for all truth parameters that are present in the samples.
    """

    parameter_names = [
        name
        for name in truth_parameters.keys()
        if name in samples.columns
    ]

    if len(parameter_names) == 0:
        raise ValueError(
            "None of the injected truth parameters are present in sampler.samples. "
            f"Truth keys: {list(truth_parameters.keys())}. "
            f"Sample columns: {list(samples.columns)}."
        )

    truth_parameters = {}

    for suffix, params in zip(
        ["_A", "_B"],
        DP.scenario.injct_params_waves,
    ):
        # Add the injection parameters as stored.
        for key, value in params.items():
            truth_parameters[f"{key}{suffix}"] = value

        # The posterior uses component masses, whereas the injection
        # is stored as chirp_mass and mass_ratio.
        if "chirp_mass" in params and "mass_ratio" in params:
            mass_1, mass_2 = chirp_q_to_component_masses(
                params["chirp_mass"],
                params["mass_ratio"],
            )

            truth_parameters[f"mass_1{suffix}"] = mass_1
            truth_parameters[f"mass_2{suffix}"] = mass_2
    
    # OLD, for full analysis
    # fig = corner.corner(
    #     samples[parameter_names].to_numpy(),
    #     labels=parameter_names,
    #     truths=truths,
    #     show_titles=True,
    # )
    # TEMP
    plot_columns = [
        "chirp_mass_A",
        "mass_ratio_A",
        "chirp_mass_B",
        "mass_ratio_B",
    ]
    fig = corner.corner(
        samples[plot_columns],
        labels=plot_columns,
        truths=[
            truth_parameters["chirp_mass_A"],
            truth_parameters["mass_ratio_A"],
            truth_parameters["chirp_mass_B"],
            truth_parameters["mass_ratio_B"],
        ],
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_file

def add_component_masses(samples):
    samples = samples.copy()

    valid = np.ones(len(samples), dtype=bool)

    for suffix in ("_A", "_B"):
        chirp_key = f"chirp_mass{suffix}"
        ratio_key = f"mass_ratio{suffix}"

        valid &= np.isfinite(samples[chirp_key])
        valid &= np.isfinite(samples[ratio_key])
        valid &= samples[chirp_key] > 0.0
        valid &= samples[ratio_key] > 0.0
        valid &= samples[ratio_key] <= 1.0

    print(
        f"Kept {valid.sum()} / {len(valid)} samples; "
        f"removed {(~valid).sum()} invalid mass samples"
    )

    samples = samples.loc[valid].copy()

    for suffix in ("_A", "_B"):
        chirp_key = f"chirp_mass{suffix}"
        ratio_key = f"mass_ratio{suffix}"

        mass_1, mass_2 = chirp_q_to_component_masses(
            samples[chirp_key].to_numpy(),
            samples[ratio_key].to_numpy(),
        )

        samples[f"mass_1{suffix}"] = mass_1
        samples[f"mass_2{suffix}"] = mass_2

    return samples

def run_minimal_symmetry_checks(samples, *, atol=1e-12):
    """Check only the conventions that remove the two relevant label swaps."""
    for suffix in ("_A", "_B"):
        q = samples[f"mass_ratio{suffix}"].to_numpy(dtype=float)
        m1 = samples[f"mass_1{suffix}"].to_numpy(dtype=float)
        m2 = samples[f"mass_2{suffix}"].to_numpy(dtype=float)

        assert np.all((q > 0.0) & (q <= 1.0 + atol)), (
            f"Invalid mass_ratio values for {suffix}: expected 0 < q <= 1."
        )
        assert np.all(m1 + atol >= m2), (
            f"Component ordering failed for {suffix}: found mass_2 > mass_1."
        )

    time_columns = {"geocent_time_A", "geocent_time_B", "delta_t_AB"}
    if time_columns.issubset(samples.columns):
        expected_time_b = (
            samples["geocent_time_A"].to_numpy(dtype=float)
            + samples["delta_t_AB"].to_numpy(dtype=float)
        )
        actual_time_b = samples["geocent_time_B"].to_numpy(dtype=float)

        assert np.all(samples["delta_t_AB"].to_numpy(dtype=float) > 0.0), (
            "Signal ordering failed: expected delta_t_AB > 0."
        )
        assert np.allclose(actual_time_b, expected_time_b, rtol=0.0, atol=atol), (
            "geocent_time_B is inconsistent with geocent_time_A + delta_t_AB."
        )

def test_A_B_analysis(result):
    df = result.copy()

    cut = 17.0

    df["mass_mode"] = np.select(
        [
            (df["chirp_mass_A"] < cut) & (df["chirp_mass_B"] < cut),
            (df["chirp_mass_A"] < cut) & (df["chirp_mass_B"] >= cut),
            (df["chirp_mass_A"] >= cut) & (df["chirp_mass_B"] < cut),
            (df["chirp_mass_A"] >= cut) & (df["chirp_mass_B"] >= cut),
        ],
        ["low-low", "low-high", "high-low", "high-high"],
        default="unclassified",
    )

    print(df["mass_mode"].value_counts(normalize=True))

    if "log_likelihood" in df:
        print(
            df.groupby("mass_mode")["log_likelihood"]
            .agg(["count", "median", "max"])
            .sort_values("max", ascending=False)
        )

def branch_table(samples):
    def get_branch(columns):
        x = samples[columns].to_numpy()

        km = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=0,
        ).fit(x)

        labels = km.labels_

        # Label branches by increasing chirp mass:
        # 0 = low chirp mass, 1 = high chirp mass
        order = np.argsort(km.cluster_centers_[:, 0])
        mapping = {
            order[0]: "low",
            order[1]: "high",
        }

        return pd.Series(labels).map(mapping)

    branch_a = get_branch(
        ["chirp_mass_A", "mass_ratio_A"]
    )
    branch_b = get_branch(
        ["chirp_mass_B", "mass_ratio_B"]
    )

    return pd.crosstab(
        branch_a,
        branch_b,
        normalize="all",
        rownames=["A branch"],
        colnames=["B branch"],
    )

def Main():
    print('test')
    model_name = "best_model.pt"
    scenario, logger, _, _ = setUpLoggerScenario()

    # If setUpLoggerScenario does not already call this, uncomment:
    # scenario.setUpScenario()

    DP = DINGO_pipeline(logger, scenario, generateData = False)

    # ----------------------------------------------------
    # Optional node-local dataset cache.
    # This changes only our wrapper settings, not Dingo.
    # ----------------------------------------------------

    use_local_cache = (
        os.environ.get(
            "GW_USE_LOCAL_CACHE",
            "0",
        )
        == "1"
    )

    if use_local_cache:

        cache_dir = os.environ.get(
            "GW_LOCAL_CACHE"
        )

        if cache_dir is None:
            raise RuntimeError(
                "GW_USE_LOCAL_CACHE=1 but "
                "GW_LOCAL_CACHE is not defined."
            )

        DP.local_settings[
            "local_cache_path"
        ] = cache_dir

    else:

        DP.local_settings.pop(
            "local_cache_path",
            None,
        )

    print(
        "USE LOCAL CACHE:",
        use_local_cache,
    )

    print(
        "local_cache_path:",
        DP.local_settings.get(
            "local_cache_path"
        ),
    )

    if DIAGNOSE_IO:
        diagnose_dingo_io(
            DP,
            batch_size=64,
            repeats=2,
        )
        return
    #test for label switching
    # logger.info("$$$ begin Testing DINGO pipeline") 
    # posterior_samples_1 = DP.infer_from_strain(
    #         num_samples=NUM_SAMPLES_TO_PLOT,
    #     )
    # posterior_samples_0 = DP.infer_from_strain(
    #         num_samples=NUM_SAMPLES_TO_PLOT,
    #         model_name=model_name #model_270_1, best_model_1
    #     )
    
    # print("15 epochs after unfreezing")
    # print(branch_table(posterior_samples_0))

    # print("\nAfter copula training")
    # print(branch_table(posterior_samples_1))
    
    if TRAIN:
        logger.info("$$$ training DINGO pipeline")
        DP.train()

    if INFER:
        logger.info("$$$ running DINGO inference on initialized scenario")

        samples = DP.infer_from_strain(
            num_samples=NUM_SAMPLES_TO_PLOT,
            model_name=model_name
        )
        
        samples = add_component_masses(samples)
        run_minimal_symmetry_checks(samples)
        test_A_B_analysis(samples)
        logger.info(
            "$$$ minimal symmetry checks passed: "
            "q in (0, 1], mass_1 >= mass_2, and positive geocent-time ordering."
        )
        
        if PLOT:
            truth_parameters = {}

            for suffix, params in zip(["_A", "_B"], DP.scenario.injct_params_waves):
                for key, value in params.items():
                    truth_parameters[f"{key}{suffix}"] = value
            
            corner_path = make_corner_plot(
                samples          = samples,
                truth_parameters = truth_parameters,
                out_file         = Path(DP.train_dir) / "scenario_inference_corner.png",
                DP               = DP
            )

            logger.info(f"$$$ wrote corner plot to {corner_path}")

if __name__ == "__main__":

    Main()
    print("End")
