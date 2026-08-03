from utils import get_dingo_dir_data
from config import ScenarioConfig
from setUpLoggerScenario import setUpLoggerScenario
from Pipeline import DINGO_pipeline
from dingo.core.result import Result, make_pp_plot
from pathlib import Path
import numpy as np
import pandas as pd
from bilby.core.prior import Prior, PriorDict
from collections.abc import Mapping

GROUP_SUFFIXES = {
    "waveFormA": "A",
    "waveFormB": "B",
}


def to_1d_numpy(values, parameter_name: str) -> np.ndarray:
    """Convert NumPy or Torch posterior samples to a finite 1D NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()

    array = np.asarray(values)
    array = np.squeeze(array)

    if array.ndim != 1:
        raise ValueError(
            f"Samples for {parameter_name!r} must be one-dimensional after "
            f"squeezing, but have shape {array.shape}."
        )

    return array

def flatten_truth(
    theta_true,
    keys: list[str],
) -> dict[str, float]:
    """
    Convert the injected A/B parameter representation into a flat dictionary.

    Supported input formats
    -----------------------
    1. List or tuple:
        [
            {"chirp_mass": ..., "mass_ratio": ...},  # A
            {"chirp_mass": ..., "mass_ratio": ...},  # B
        ]

    2. Nested dictionary:
        {
            "waveFormA": {"chirp_mass": ..., "mass_ratio": ...},
            "waveFormB": {"chirp_mass": ..., "mass_ratio": ...},
        }

    Output
    ------
        {
            "chirp_mass_A": float(...),
            "mass_ratio_A": float(...),
            "chirp_mass_B": float(...),
            "mass_ratio_B": float(...),
        }
    """

    # Convert a two-element list into the expected named structure.
    if isinstance(theta_true, (list, tuple)):
        if len(theta_true) != 2:
            raise ValueError(
                "Expected theta_true to contain exactly two waveform "
                f"parameter dictionaries, but received {len(theta_true)}."
            )

        theta_true = {
            "waveFormA": theta_true[0],
            "waveFormB": theta_true[1],
        }

    if not isinstance(theta_true, Mapping):
        raise TypeError(
            "theta_true must be either a mapping or a two-element list/tuple, "
            f"not {type(theta_true).__name__}."
        )

    flattened = {}

    for group_name, suffix in GROUP_SUFFIXES.items():
        if group_name not in theta_true:
            raise KeyError(
                f"Truth input has no {group_name!r} entry. "
                f"Available entries: {list(theta_true)}"
            )

        parameter_dict = theta_true[group_name]

        if not isinstance(parameter_dict, Mapping):
            raise TypeError(
                f"theta_true[{group_name!r}] must be a dictionary, "
                f"not {type(parameter_dict).__name__}."
            )

        for parameter_name, value in parameter_dict.items():
            # Avoid adding a suffix twice.
            if parameter_name.endswith(f"_{suffix}"):
                flat_name = parameter_name
            else:
                flat_name = f"{parameter_name}_{suffix}"

            # Only retain parameters requested for the P–P plot.
            if flat_name not in keys:
                continue

            if hasattr(value, "detach"):
                value = value.detach().cpu()

                if value.numel() != 1:
                    raise ValueError(
                        f"Injected value for {flat_name!r} must be scalar, "
                        f"but has tensor shape {tuple(value.shape)}."
                    )

                value = value.item()

            value_array = np.asarray(value)

            if value_array.size != 1:
                raise ValueError(
                    f"Injected value for {flat_name!r} must be scalar, "
                    f"but has shape {value_array.shape}."
                )

            # Dingo checks isinstance(value, float), so force a Python float.
            flattened[flat_name] = float(value_array.item())

    missing = set(keys) - set(flattened)

    if missing:
        raise KeyError(
            "Injected truth is missing the following P–P parameters: "
            f"{sorted(missing)}. Flattened parameters were "
            f"{sorted(flattened)}."
        )

    return flattened

def posterior_to_dataframe(
    posterior_samples,
    keys: list[str],
) -> pd.DataFrame:
    """
    Accept one of:

    1. A flat DataFrame with columns such as chirp_mass_A.
    2. A flat dict mapping parameter names to sample arrays.
    3. A nested dict:
           {
               "waveFormA": DataFrame or dict,
               "waveFormB": DataFrame or dict,
           }
    """
    if isinstance(posterior_samples, pd.DataFrame):
        dataframe = posterior_samples.copy()

    elif isinstance(posterior_samples, Mapping):
        is_nested = any(
            group_name in posterior_samples
            for group_name in GROUP_SUFFIXES
        )

        if not is_nested:
            # Flat dict:
            # {"chirp_mass_A": samples, "mass_ratio_A": samples, ...}
            dataframe = pd.DataFrame(
                {
                    name: to_1d_numpy(values, name)
                    for name, values in posterior_samples.items()
                }
            )

        else:
            columns = {}

            for group_name, suffix in GROUP_SUFFIXES.items():
                if group_name not in posterior_samples:
                    raise KeyError(
                        f"Posterior output has no {group_name!r} entry. "
                        f"Available entries: {list(posterior_samples)}"
                    )

                group_samples = posterior_samples[group_name]

                if isinstance(group_samples, pd.DataFrame):
                    iterator = group_samples.items()
                elif isinstance(group_samples, Mapping):
                    iterator = group_samples.items()
                else:
                    raise TypeError(
                        f"posterior_samples[{group_name!r}] must be a "
                        f"DataFrame or dict, not "
                        f"{type(group_samples).__name__}."
                    )

                for parameter_name, values in iterator:
                    flat_name = (
                        parameter_name
                        if parameter_name.endswith(f"_{suffix}")
                        else f"{parameter_name}_{suffix}"
                    )

                    if flat_name in keys:
                        columns[flat_name] = to_1d_numpy(
                            values,
                            flat_name,
                        )

            dataframe = pd.DataFrame(columns)

    else:
        raise TypeError(
            "posterior_samples must be a pandas DataFrame or dictionary, "
            f"not {type(posterior_samples).__name__}."
        )

    missing = set(keys) - set(dataframe.columns)

    if missing:
        raise KeyError(
            f"Posterior output is missing the following columns: "
            f"{sorted(missing)}. Available columns: "
            f"{list(dataframe.columns)}"
        )

    dataframe = (
        dataframe.loc[:, keys]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    if dataframe.empty:
        raise ValueError("No finite posterior samples remain.")

    return dataframe


def make_calibration_result(
    posterior_samples,
    theta_true,
    keys: list[str],
    plot_prior: PriorDict,
) -> Result:
    """Build one minimal Dingo Result for one calibration event."""
    posterior_df = posterior_to_dataframe(
        posterior_samples=posterior_samples,
        keys=keys,
    )
    
    # This conversion must not be commented out.
    injection_parameters = flatten_truth(
        theta_true=theta_true,
        keys=keys,
    )

    if not isinstance(injection_parameters, dict):
        raise TypeError(
            "flatten_truth must return a dictionary, "
            f"but returned {type(injection_parameters).__name__}."
        )

    result = Result(
        dictionary={
            "samples": posterior_df,
            "event_metadata": {
                # Use the flattened dictionary, not theta_true.
                "injection_parameters": injection_parameters,
            },
        }
    )

    result.prior = plot_prior

    # Fail here rather than after all inference runs have finished.
    if not isinstance(result.injection_parameters, dict):
        raise TypeError(
            "The constructed Result has non-dictionary injection parameters: "
            f"{type(result.injection_parameters).__name__}."
        )

    return result

def Main():
    n_events            = 100
    n_posterior_samples = 5000
    
    keys=[
        "chirp_mass_A",
        "mass_ratio_A",
        "chirp_mass_B",
        "mass_ratio_B",
    ]
    
    ScenConfig                 = ScenarioConfig()
    scenario,logger,_,data_dir = setUpLoggerScenario(ScenConfig)
    dataPipeline               = DINGO_pipeline(logger,scenario,False) # read from Dingo/data/training_run/model_latest.pt
    plot_prior                 = scenario.prior.getJointPriors()
    
    results = []
    for i in range(n_events):
        logger.info(
            "Generating calibration event %d/%d.",
            i + 1,
            n_events,
        )
        theta_true  = scenario.prior.GetWaveFormParamsSampled()  #dict of 2 events listed
        context     = dataPipeline.build_context_from_initialized_scenario(theta_true)
        posterior_samples = dataPipeline.infer(
            num_samples = n_posterior_samples,
            injection   = context
        )
        
        result = make_calibration_result(
            posterior_samples=posterior_samples,
            theta_true=theta_true,
            keys=keys,
            plot_prior=plot_prior,
        ) 
        
        results.append(result)
    output_directory = Path(data_dir) / "calibration"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "pp_plot.pdf"

    # This must receive the complete list of results, not one Result.
    fig, pvals = make_pp_plot(
        results=results,
        keys=keys,
        filename=str(output_file),
        save=True,
        weighted=False,
    )
    return fig, pvals, results

###########################
###   Run actual Code   ###
###########################

if __name__ == "__main__":
    #TODO: remove
    Main()