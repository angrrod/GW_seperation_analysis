import bilby
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import os
import json
import corner
from matplotlib.lines import Line2D
from scenario import ScenarioConfig
from scenario import GWScenario
from methods import Method_type, MethodConfig


### helper functions ###
def SaveResults(results, out_dir="out", filename="results.json"):
    """Write a dictionary to a JSON file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return path

def plotOverlap(params,results,truths,logger):
    logger.info("$$$ Making corner plots")
    
    out_dir = "out"
    plot_dir = os.path.join(out_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig                = None  #necessary for initialization
    params_A, params_B = addSuffixes(params)
    
    #colors
    cmap               = plt.get_cmap("tab20")
    colors             = list(cmap.colors)        # length 20
    n_colors           = len(colors)
    color_idx          = 0 
    legend_handles = []  
    legend_labels  = []
    
    for method in results:
        res = results[method]
        for waveform in res:
            wave = res[waveform]
            color = colors[color_idx % n_colors]
            color_idx += 1            #separate the joint poisterior that ends with _A and _B in their respective posterior samples
            label = f"{method.code} – {waveform}"
            
            # if method == Method_type.JOINT:
            #     df_A = wave.posterior.copy()
            #     df_A.rename(columns=params_A, inplace=True)
            #     df_B = wave.posterior.copy()
            #     df_B.rename(columns=params_B, inplace=True)
            #     fig = createCornerPlot(df_A[params].values,params,color,fig,truths[1])  #index doesn't matter as things get overlapped
            #     fig = createCornerPlot(df_B[params].values,params,color,fig,truths[0])
            # else:
                #Single waveform
            fig = createCornerPlot(wave.posterior[params].values,params,color,fig,truths[0])
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
    
def createCornerPlot(samps,params,color,fig,truths):
    fig = corner.corner(
        samps,
        labels        = params,          # base labels, no _A/_B
        color         = color,
        truths        = truths,
        truth_color   = "black",
        plot_contours = True,
        fill_contours = False,
        hist_kwargs   = dict(density=True),
        fig           = fig,    # None for first call; existing fig later
    )
    return fig

def addSuffixes(strings):
    suffixed_A = {s + "_A":s for s in strings}
    suffixed_B = {s + "_B":s for s in strings}
    return suffixed_A, suffixed_B


###########################
####     Main Loop     ####
###########################

def Main(run_sampler,saveResults):
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    # needed for multi threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    #set-up plotting dirs
    out_dir = "out"
    plot_dir = os.path.join(out_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    #Logger
    bilby.core.utils.setup_logger(
        log_level="INFO",
        label="my_run", 
        outdir="out",
    )
    logger = bilby.core.utils.logger
    logger.info("$$$ start_run")
    
    #build scenario
    ScenConfig     = ScenarioConfig()
    scenario       = GWScenario(logger, ScenConfig)
    scenario.setUpScenario()
    scenario.makePlots(["strain_time_domain_set_up","qtransform_set_up"])
    
    results       = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for measuring overlap etc with the joint.
    samplesToPlot = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for plotting
    MethodConf    = MethodConfig()
    
    for method_type in Method_type:
        logger.info(f"$$$ Running method: {method_type.code}")
        method  = method_type.method(run_sampler,scenario,logger,MethodConf)
        
        start                           = time.process_time()
        method.generateSamples()
        end                             = time.process_time()
        runTime                         = end - start
        results[method_type.code]['runTime'] = runTime
        # if method_type == Method_type.JOINT:
        #     JointBaselineDistr
        # elif JointBaselineDistr is None:
        #     raise ValueError("joint serves as a baseline and needs to be ran first")
        
        # else: pass
        
        #plotting info
        sample                          = method.posteriors
        samplesToPlot[method_type]      = sample
        
        #build measurements
        diagnostics = {}
        diagnostics["information_gain"] = []
        for waveform in sample.keys():
            diagnostics["information_gain"].append(sample[waveform].information_gain) #KL between prior and posterior.
        results[method_type.code]['diagnostics'] = diagnostics
        
        # analyze the samples
    
    #post processing
    # injct_params_wave = sample.to_dict(orient="records")[0:1]
    # SetupSignalAndDetector(wfv_args,ASD_file_name,injct_params_wave,True,logger)
    if saveResults:
        SaveResults(results)
        
    #make corner plot of the posterior samples
    params = [
        "chirp_mass",
        "mass_ratio", 
        "luminosity_distance",
        "psi",
        "phase",
        "geocent_time",
        # "ra",
        # "dec"
        ] #,"mass_ratio","luminosity_distance"
    truths = scenario.GetWaveFormParams()
    truths = [[d[k] for k in params if k in d] for d in truths]  #get params to be plotted in corner plot
    plotOverlap(params,samplesToPlot,truths,logger)
    
###########################
###   Run actual Code   ###
###########################
Main(run_sampler = True, saveResults = True)