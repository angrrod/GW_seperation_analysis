### this script is made so it analyses the results obtained from the main GW sampeling step ###
from __future__ import annotations

import utils
from config import ScenarioConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from Pipeline import Pipeline_type
from setUpLoggerScenario import setUpLoggerScenario
import pandas as pd
import numpy as np

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr


def compare_waveform_dataframes(
    data: dict[str, pd.DataFrame],
    key_a: str = "waveFormA",
    key_b: str = "waveFormB",
    tail_quantile: float = 0.95,
    min_observations: int = 10,
    alignment: Literal["index", "position"] = "index",
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Compare corresponding columns from two DataFrames stored in a dictionary.

    The default expected structure is:

        data = {
            "waveFormA": dataframe_a,
            "waveFormB": dataframe_b,
        }

    Columns with the same name in both DataFrames are compared.

    Parameters
    ----------
    data
        Dictionary containing the two DataFrames.

    key_a, key_b
        Dictionary keys corresponding to the A and B DataFrames.

    tail_quantile
        Quantile threshold for empirical tail dependence.

        For example, 0.95 examines whether observations jointly occur
        in the upper or lower 5% of their marginal distributions.

    min_observations
        Minimum number of valid paired observations required.

    alignment
        How rows from the two DataFrames should be paired:

        - "index":
          Match observations by their DataFrame index. This is appropriate
          when both DataFrames originated from the same observations and
          retained their original indices.

        - "position":
          Match the first row of A with the first row of B, the second with
          the second, and so on. Both DataFrames must have the same length.

    print_results
        Whether to print the resulting table.

    Returns
    -------
    pd.DataFrame
        One row per matched variable, containing:

        - Pearson correlation
        - Spearman's rho
        - Kendall's tau
        - empirical lower-tail dependence
        - empirical upper-tail dependence
        - associated p-values for the correlation coefficients
    """
    if key_a not in data:
        raise KeyError(f"Dictionary does not contain the key {key_a!r}.")

    if key_b not in data:
        raise KeyError(f"Dictionary does not contain the key {key_b!r}.")

    dataframe_a = data[key_a]
    dataframe_b = data[key_b]

    if not isinstance(dataframe_a, pd.DataFrame):
        raise TypeError(f"data[{key_a!r}] must be a pandas DataFrame.")

    if not isinstance(dataframe_b, pd.DataFrame):
        raise TypeError(f"data[{key_b!r}] must be a pandas DataFrame.")

    if not 0.5 < tail_quantile < 1.0:
        raise ValueError(
            "`tail_quantile` must lie strictly between 0.5 and 1.0."
        )

    if min_observations < 2:
        raise ValueError("`min_observations` must be at least 2.")

    if alignment not in {"index", "position"}:
        raise ValueError(
            "`alignment` must be either 'index' or 'position'."
        )

    common_columns = sorted(
        set(dataframe_a.columns).intersection(dataframe_b.columns)
    )

    if not common_columns:
        raise ValueError(
            f"No common columns were found between data[{key_a!r}] "
            f"and data[{key_b!r}]."
        )

    only_in_a = sorted(set(dataframe_a.columns) - set(dataframe_b.columns))
    only_in_b = sorted(set(dataframe_b.columns) - set(dataframe_a.columns))

    if only_in_a:
        warnings.warn(
            f"Columns present only in {key_a!r} will be ignored: "
            f"{only_in_a}",
            stacklevel=2,
        )

    if only_in_b:
        warnings.warn(
            f"Columns present only in {key_b!r} will be ignored: "
            f"{only_in_b}",
            stacklevel=2,
        )

    results = []

    for column in common_columns:
        x, y = _extract_paired_values(
            dataframe_a=dataframe_a,
            dataframe_b=dataframe_b,
            column=column,
            alignment=alignment,
        )

        valid = np.isfinite(x) & np.isfinite(y)

        x_valid = x[valid]
        y_valid = y[valid]

        n_total = len(x)
        n_valid = len(x_valid)
        n_removed = n_total - n_valid

        result = {
            "variable": column,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_removed": n_removed,
            "pearson_r": np.nan,
            "pearson_pvalue": np.nan,
            "spearman_rho": np.nan,
            "spearman_pvalue": np.nan,
            "kendall_tau": np.nan,
            "kendall_pvalue": np.nan,
            "lower_tail_dependence": np.nan,
            "upper_tail_dependence": np.nan,
        }

        if n_valid < min_observations:
            warnings.warn(
                f"Cannot reliably compare column {column!r}: "
                f"only {n_valid} valid paired observations are available.",
                stacklevel=2,
            )
            results.append(result)
            continue

        x_is_constant = np.all(x_valid == x_valid[0])
        y_is_constant = np.all(y_valid == y_valid[0])

        if x_is_constant or y_is_constant:
            warnings.warn(
                f"Correlation is undefined for column {column!r} because "
                "at least one of the two variables is constant.",
                stacklevel=2,
            )
            results.append(result)
            continue

        pearson_result = pearsonr(x_valid, y_valid)
        spearman_result = spearmanr(x_valid, y_valid)
        kendall_result = kendalltau(x_valid, y_valid)

        lower_tail, upper_tail = _empirical_tail_dependence(
            x=x_valid,
            y=y_valid,
            quantile=tail_quantile,
        )

        result.update(
            {
                "pearson_r": pearson_result.statistic,
                "pearson_pvalue": pearson_result.pvalue,
                "spearman_rho": spearman_result.statistic,
                "spearman_pvalue": spearman_result.pvalue,
                "kendall_tau": kendall_result.statistic,
                "kendall_pvalue": kendall_result.pvalue,
                "lower_tail_dependence": lower_tail,
                "upper_tail_dependence": upper_tail,
            }
        )

        results.append(result)

    output = pd.DataFrame(results)

    if print_results:
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            200,
            "display.precision",
            4,
        ):
            print(output.to_string(index=False))

    return output

def _extract_paired_values(
    dataframe_a: pd.DataFrame,
    dataframe_b: pd.DataFrame,
    column: str,
    alignment: Literal["index", "position"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a pair of numeric arrays using index or positional alignment.
    """
    if alignment == "index":
        paired = pd.concat(
            [
                dataframe_a[column].rename("A"),
                dataframe_b[column].rename("B"),
            ],
            axis=1,
            join="inner",
        )

        x = pd.to_numeric(
            paired["A"],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            paired["B"],
            errors="coerce",
        ).to_numpy(dtype=float)

        return x, y

    if len(dataframe_a) != len(dataframe_b):
        raise ValueError(
            "Positional alignment requires both DataFrames to have the "
            f"same number of rows, but they contain {len(dataframe_a)} "
            f"and {len(dataframe_b)} rows."
        )

    x = pd.to_numeric(
        dataframe_a[column].reset_index(drop=True),
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        dataframe_b[column].reset_index(drop=True),
        errors="coerce",
    ).to_numpy(dtype=float)

    return x, y

def _empirical_tail_dependence(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float,
) -> tuple[float, float]:
    """
    Calculate finite-threshold empirical lower- and upper-tail dependence.

    These are empirical estimates evaluated at a particular quantile,
    rather than estimates of the asymptotic tail-dependence coefficients.
    """
    n = len(x)

    # Convert both variables to pseudo-observations on (0, 1).
    # Average ranks provide a reasonable treatment of tied values.
    u = rankdata(x, method="average") / (n + 1.0)
    v = rankdata(y, method="average") / (n + 1.0)

    lower_threshold = 1.0 - quantile
    upper_threshold = quantile
    tail_probability = 1.0 - quantile

    lower_joint_probability = np.mean(
        (u <= lower_threshold) & (v <= lower_threshold)
    )

    upper_joint_probability = np.mean(
        (u >= upper_threshold) & (v >= upper_threshold)
    )

    lower_tail_dependence = (
        lower_joint_probability / tail_probability
    )

    upper_tail_dependence = (
        upper_joint_probability / tail_probability
    )

    return lower_tail_dependence, upper_tail_dependence

def Main(useJoint = True,onlyCorr = True):
    #build scenario
    ScenConfig     = ScenarioConfig()
    scenario,logger,plot_dir,data_dir = setUpLoggerScenario(ScenConfig)
    data_dir = data_dir/ "results_joint_final.hdf5"
    results = utils.exctractResults(data_dir,logger)
    
    parameter_name = 'mass_ratio'#'mass_ratio' #dec luminosity_distance chirp_mass geocent_time phase
    posteriors = results['joint_likl']['posteriors']
    fig_cor = plot_joint_parameter_corner(posteriors = posteriors,parameter = parameter_name)
    path = os.path.join(plot_dir, "correlation.png")
    fig_cor.savefig(path, dpi=200)
    compare_waveform_dataframes(posteriors)
    
    if not onlyCorr:
        # t1 = results['joint_likl']['posteriors']['waveFormA']['chirp_mass']
        # t2 = results['joint_likl']['posteriors']['waveFormA']['geocent_time']
        #modify the results
        #KL between joint posterior and others posterior.
        if useJoint:
            for method in results.keys():
                if method != 'joint_likl':
                    KL_res,KL_res_rev = utils.calculate_KL_between_joint(results,method,logger)
                    results[method]['diagnostics']['methodInfo']['KL_div_avg']     = KL_res
                    results[method]['diagnostics']['methodInfo']['KL_div_rev_avg'] = KL_res_rev
        else:
            #remove joint value
            results.pop('joint_likl',None)
        
        # make corner plot of the posterior samples
        scenario.makePlots(["strain_time_domain_set_up","qtransform_set_up"],plot_dir)
        params = [
            "chirp_mass",
            "mass_ratio", 
            # "psi",
            # "phase",
            # "geocent_time",
            # "luminosity_distance"
            # "ra",
            # "dec"
            ] #,"mass_ratio","luminosity_distance"
        truths = scenario.prior.GetWaveFormParamsFixed()
        truths = [[d[k] for k in params if k in d] for d in truths]  #get params to be plotted in corner plot
        plotOverlap(params,results,truths,logger,plot_dir)

def plotOverlap(params,results,truths,logger,plot_dir):
    logger.info("$$$ Making corner plots")
    
    fig                = None  #necessary for initialization
    params_A, params_B = utils.addSuffixes(params)
    
    #colors
    cmap               = plt.get_cmap("tab20")
    colors             = list(cmap.colors)        # length 20
    n_colors           = len(colors)
    color_idx          = 0 
    legend_handles = []  
    legend_labels  = []
    
    for pipeline in results:
        if pipeline != Pipeline_type.SINGLE.code:
            res = results[pipeline].get('posteriors')
            for waveform in res:
                wave = res[waveform]
                color = colors[color_idx % n_colors]
                color_idx += 1            #separate the joint poisterior that ends with _A and _B in their respective posterior samples
                label = f"{pipeline} – {waveform}"
                
                if pipeline == Pipeline_type.JOINT.code:
                    df_A = wave.copy()
                    df_A.rename(columns=params_A, inplace=True)
                    df_B = wave.copy()
                    df_B.rename(columns=params_B, inplace=True)
                    fig = utils.createCornerPlot(df_A[params].values,params,color,fig,truths[1])  #index doesn't matter as things get overlapped
                    fig = utils.createCornerPlot(df_B[params].values,params,color,fig,truths[0])
                else:
                    #Single waveform
                    fig = utils.createCornerPlot(wave[params].values,params,color,fig,truths[0])
                legend_handles.append(
                    Line2D([0], [0], color=color, lw=2)
                )
                legend_labels.append(label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout()
    
    fileName = "multiple waveforms"
    path = os.path.join(plot_dir, f"{fileName}.png")
    fig.savefig(path, dpi=200)

def plot_joint_parameter_corner(
    posteriors: dict,
    parameter: str,
    color: str = "C0",
    fig=None,
    truths=None,
    waveform_a_key: str = "waveFormA",
    waveform_b_key: str = "waveFormB",
    strict_index_check: bool = False,
):
    """
    Plot p(parameter_A, parameter_B) as a corner plot.

    The pairing is row-wise:
        sample i from waveFormA is paired with sample i from waveFormB.

    This preserves the sample order from the stored posterior tables.
    """

    df_A = posteriors[waveform_a_key]
    df_B = posteriors[waveform_b_key]

    if parameter not in df_A.columns:
        raise KeyError(f"{parameter!r} not found in {waveform_a_key}")

    if parameter not in df_B.columns:
        raise KeyError(f"{parameter!r} not found in {waveform_b_key}")

    if len(df_A) != len(df_B):
        raise ValueError(
            f"Cannot make joint plot: {waveform_a_key} has {len(df_A)} samples, "
            f"but {waveform_b_key} has {len(df_B)} samples."
        )

    if strict_index_check and not df_A.index.equals(df_B.index):
        raise ValueError(
            f"{waveform_a_key} and {waveform_b_key} do not have matching indices. "
            "If row order is still meaningful, call with strict_index_check=False."
        )

    samples = np.column_stack(
        [
            df_A[parameter].to_numpy(),
            df_B[parameter].to_numpy(),
        ]
    )

    finite_mask = np.isfinite(samples).all(axis=1)
    samples = samples[finite_mask]

    labels = [f"{parameter}_A", f"{parameter}_B"]

    return utils.createCornerPlot(
        samps=samples,
        params=labels,
        color=color,
        fig=fig,
        truths=truths,
    )    

###########################
###   Run actual Code   ###
###########################
if __name__ == "__main__":
    Main()