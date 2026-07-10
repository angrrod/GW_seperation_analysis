### this script is made so it analyses the results obtained from the main GW sampeling step ###
import utils
from config import ScenarioConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from Pipeline import Pipeline_type
from setUpLoggerScenario import setUpLoggerScenario
import pandas as pd
import numpy as np

def Main(useJoint = True):
    #build scenario
    ScenConfig     = ScenarioConfig()
    scenario,logger,plot_dir,data_dir = setUpLoggerScenario(ScenConfig)
    data_dir = data_dir/ "results.hdf5"
    results = utils.exctractResults(data_dir,logger)
    
    parameter_name = 'luminosity_distance'#'mass_ratio' #dec
    fig_cor = plot_joint_parameter_corner(posteriors = results['joint_likl']['posteriors'],parameter = parameter_name)
    path = os.path.join(plot_dir, "correlation.png")
    fig_cor.savefig(path, dpi=200)
    
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
        "geocent_time",
        "luminosity_distance"
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