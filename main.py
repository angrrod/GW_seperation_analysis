import time
from collections import defaultdict
import os
from methods import Method_type, MethodConfig
import utils



###########################
####     Main Loop     ####
###########################

def Main(run_sampler):
    """_summary_
    Args:
        run_sampler (bool): Describes if the sampler should be run from scratch, performing an entire sampeling run.
    """
    # needed for multi threading
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    scenario,logger,_,data_dir = utils.setUpLoggerScenario()
    
    results       = defaultdict(dict, {mt.code: {} for mt in Method_type}) #used for measuring overlap etc with the joint.
    MethodConf    = MethodConfig()
    
    for method_type in Method_type:
        logger.info(f"$$$ Running method: {method_type.code}")
        method  = method_type.method(run_sampler,scenario,logger,MethodConf)
        
        start                                = time.process_time()
        method.generateSamples()
        end                                  = time.process_time()
        runTime                              = end - start
        results[method_type.code]['runTime'] = runTime
        method_meta = {"runTime" : runTime}
        
        utils.writeMethodResult(data_dir,method_type.code,method_meta,method.results,logger)        
        
        
###########################
###   Run actual Code   ###
###########################

Main(run_sampler = False)