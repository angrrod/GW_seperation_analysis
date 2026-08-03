
from config.ScenarioConfig import ScenarioConfig
from Pipeline import DINGO_pipeline
from setUpLoggerScenario import setUpLoggerScenario
import corner
import matplotlib.pyplot as plt
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.base_model import BasePosteriorModel
from pathlib import Path
import os
import numpy as np


###########################
###   Run Debug Code  e  ###
###########################

TRAIN                 = False
INFER                 = True
PLOT                  = True
GENERATE_DATA         = False
NUM_SAMPLES_TO_PLOT   = 20000

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

def Main():
    print('test')
    ScenConfig = ScenarioConfig()

    scenario, logger, _, _ = setUpLoggerScenario(ScenConfig)

    # If setUpLoggerScenario does not already call this, uncomment:
    # scenario.setUpScenario()

    DP = DINGO_pipeline(logger, scenario, generateData = GENERATE_DATA)

    if TRAIN:
        logger.info("$$$ training DINGO pipeline")
        DP.train()

    if INFER:
        logger.info("$$$ running DINGO inference on initialized scenario")

        samples = DP.infer_from_strain(
            num_samples=NUM_SAMPLES_TO_PLOT,
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
    # project_dir = Path(os.environ["VSC_DATA"]) / "GW_separation"
    # for directory in (
    #     project_dir / "Dingo" / "configs",
    #     project_dir / "Dingo" / "data",
    #     project_dir / "Dingo" / "logs",
    # ):
    #     directory.mkdir(parents=True, exist_ok=True)
        
    Main()
    print("End")
