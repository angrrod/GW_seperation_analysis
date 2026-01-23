### this script is made so it analyses the results obtained from the main GW sampeling step ###
import utils
from scenario import ScenarioConfig
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
from Pipeline import Pipeline_type

def Main():
    #build scenario
    ScenConfig     = ScenarioConfig()
    scenario,logger,plot_dir,data_dir = utils.setUpLoggerScenario(ScenConfig)
    data_dir = data_dir+"/results.hdf5"
    results = utils.exctractResults(data_dir,logger)
    
    #modify the results
    #KL between joint posterior and others posterior.
    for method in results.keys():
        if method != 'joint_likl':
            KL_res,KL_res_rev = utils.calculate_KL_between_joint(results,method,logger)
            results[method]['diagnostics']['methodInfo']['KL_div_avg']     = KL_res
            results[method]['diagnostics']['methodInfo']['KL_div_rev_avg'] = KL_res_rev
    
    # make corner plot of the posterior samples
    scenario.makePlots(["strain_time_domain_set_up","qtransform_set_up"],plot_dir)
    params = [
        "chirp_mass",
        "mass_ratio", 
        "psi",
        "phase",
        "geocent_time",
        "luminosity_distance",
        "ra",
        "dec"
        ] #,"mass_ratio","luminosity_distance"
    truths = scenario.GetWaveFormParams()
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
    
###########################
###   Run actual Code   ###
###########################
if __name__ == "__main__":
    Main()