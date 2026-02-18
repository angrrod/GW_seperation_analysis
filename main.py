import time
from collections import defaultdict
from methods import RunMode
from scenario import ScenarioConfig
import argparse
from Pipeline import Pipeline_type
import os
from setUpLoggerScenario import setUpLoggerScenario
###########################
####     Main Loop     ####
###########################

def Main(mode: RunMode, pipeline_type,RunDiagnostics = True):
    #start_from_chekpt only used for continuing when crash,has happend
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    # needed for multi threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    ScenConfig                 = ScenarioConfig()
    results                    = defaultdict(dict, {mt.code: {} for mt in Pipeline_type}) #used for measuring overlap etc with the joint.S
    
    scenario,logger,_,data_dir = setUpLoggerScenario(ScenConfig)
    
    if pipeline_type is None:
        raise ValueError(f"Unknown method '{pipeline_type}'")
    else:
        logger.info(f"$$$ Running method: {pipeline_type.code}")
        dataPipeline  = pipeline_type.type(logger,scenario)
        
        start                                  = time.process_time() #in seconds
        simulation_results                     = dataPipeline.run(mode)
        end                                    = time.process_time() #in seconds
        runTime                                = end - start
        results[pipeline_type.code]['runTime'] = runTime
        method_meta                            = {"runTime" : runTime}
        
        if (pipeline_type == Pipeline_type.SINGLE or pipeline_type == Pipeline_type.JOINT) and RunDiagnostics == True:
            dataPipeline.log_diagnostic_tests(simulation_results["waveFormA"],ifos_override = None)
            if pipeline_type == Pipeline_type.JOINT:
                dataPipeline.log_diagnostic_tests(simulation_results["waveFormB"],ifos_override = None)
        dataPipeline.writeMethodResult(data_dir,method_meta,simulation_results)
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GW separation pipeline for one or all methods."
    )
    parser.add_argument(
        "--run-mode",
        type=parse_run_mode,
        default=RunMode.RUN,
        help=(
            "Execution mode for samplers. "
            "Choices: run (default), reuse, checkpoint."
        ),
    )
    # New option: run a single method by code
    parser.add_argument(
        "--method",
        type=parse_pipeline_type,
        default=None,
        help="Run only one method (by pipeline_type.code). If omitted, runs all methods.",
    )
    return parser.parse_args()

def parse_pipeline_type(s: str) -> Pipeline_type:
    s = s.strip()
    for mt in Pipeline_type:
        if mt.value == s:
            return mt
    valid = [f"{mt.name} ({mt.value})" for mt in Pipeline_type]
    raise argparse.ArgumentTypeError(
        f"Unknown method '{s}'. Valid methods: {', '.join(valid)}"
    )

def parse_run_mode(s: str) -> RunMode:
    s = s.strip()
    for mode in RunMode:
        if mode.value == s:
            return mode
    valid = [mode.value for mode in RunMode]
    raise argparse.ArgumentTypeError(
        f"Unknown run mode '{s}'. Valid modes: {', '.join(valid)}"
    )
    
###########################
###   Run actual Code   ###
###########################

if __name__ == "__main__":
    #TODO: remove
    # mp.set_start_method("spawn", force=True)
    args = parse_args()
    Main(args.run_mode,args.method)