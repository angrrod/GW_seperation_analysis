
from config.ScenarioConfig import ScenarioConfig
from Pipeline import DINGO_pipeline
from setUpLoggerScenario import setUpLoggerScenario
import corner
import matplotlib.pyplot as plt
from dingo.core.posterior_models.normalizing_flow import NormalizingFlowPosteriorModel
from dingo.core.posterior_models.base_model import BasePosteriorModel
from pathlib import Path
import os



###########################
###   Run Debug Code   ###
###########################

TRAIN         = True
INFER         = True
PLOT          = True
GENERATE_DATA = True
NUM_SAMPLES   = 1000

def make_corner_plot(samples, truth_parameters, out_file):
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

    truths = [
        truth_parameters[name]
        for name in parameter_names
    ]
    
    # OLD, for full analysis
    # fig = corner.corner(
    #     samples[parameter_names].to_numpy(),
    #     labels=parameter_names,
    #     truths=truths,
    #     show_titles=True,
    # )
    # TEMP
    plot_columns = [
        "mass_1_A",
        "mass_2_A",
        "mass_1_B",
        "mass_2_B",
    ]
    fig = corner.corner(
        samples[plot_columns],
        labels=plot_columns,
        truths=[
            truth_parameters["mass_1_A"],
            truth_parameters["mass_2_A"],
            truth_parameters["mass_1_B"],
            truth_parameters["mass_2_B"],
        ],
    )

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_file

def Main():
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

        samples = DP.infer(
            num_samples=NUM_SAMPLES,
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
