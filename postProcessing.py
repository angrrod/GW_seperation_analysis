### this script is made so it analyses the results obtained from the main GW sampeling step ###
import utils
from scenario import ScenarioConfig


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
    utils.plotOverlap(params,results,truths,logger,plot_dir)
    
    
###########################
###   Run actual Code   ###
###########################
if __name__ == "__main__":
    Main()