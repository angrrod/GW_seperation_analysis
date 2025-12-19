import time
from collections import defaultdict
import os
from methods import Method_type, MethodConfig
import utils
from scenario import ScenarioConfig
import argparse


###########################
####     Main Loop     ####
###########################

def Main(run_sampler: bool, method_type):
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    # needed for multi threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    ScenConfig                 = ScenarioConfig()
    MethodConf                 = MethodConfig()
    results                    = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for measuring overlap etc with the joint.S
    
    scenario,logger,_,data_dir = utils.setUpLoggerScenario(ScenConfig)
    
    if method_type is None:
        raise ValueError(f"Unknown method '{method_type}'")
    else:
        logger.info(f"$$$ Running method: {method_type.code}")
        method  = method_type.method(run_sampler,scenario,logger,MethodConf)
        
        start                                = time.process_time() #in seconds
        method.generateSamples()
        end                                  = time.process_time() #in seconds
        runTime                              = end - start
        results[method_type.code]['runTime'] = runTime
        method_meta = {"runTime" : runTime}
        
        utils.writeMethodResult(data_dir,method_type.code,method_meta,method.results,logger)
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GW separation pipeline for one or all methods."
    )
    parser.add_argument(
        "--run-sampler",
        action="store_true",
        help="Run the sampler from scratch (default: False).",
    )
    # New option: run a single method by code
    parser.add_argument(
        "--method",
        type=parse_method_type,
        default=None,
        help="Run only one method (by Method_type.code). If omitted, runs all methods.",
    )
    return parser.parse_args()

def parse_method_type(s: str) -> Method_type:
    s = s.strip()
    for mt in Method_type:
        if mt.value == s:
            return mt
    valid = [f"{mt.name} ({mt.value})" for mt in Method_type]
    raise argparse.ArgumentTypeError(
        f"Unknown method '{s}'. Valid methods: {', '.join(valid)}"
    )
    
###########################
###   Run actual Code   ###
###########################

if __name__ == "__main__":
    args = parse_args()
    Main(args.run_sampler,args.method)