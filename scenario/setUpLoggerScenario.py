import numpy as np
import os
import bilby
from utils import get_postprocessing_dir,get_base_log_dir
from .GWScenario import GWScenario

def setUpLoggerScenario():
    #set-up plotting dirs
    log_dir = get_base_log_dir()
    out_dir = get_postprocessing_dir()
    
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_dir = os.path.join(out_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    #Logger
    bilby.core.utils.setup_logger(
        log_level="INFO", #DEBUG
        label="my_run", 
        outdir=log_dir,
    )
    logger = bilby.core.utils.logger
    logger.info("$$$ start_run")
    
    scenario   = GWScenario(logger)
    scenario.setUpScenario()
    
    #test ifo's
    for ifo in scenario.ifos:
        td = ifo.strain_data.time_domain_strain
        scenario.logger.info(f"{ifo.name}: td finite={np.isfinite(td).all()}, std={np.std(td):.3e}, maxabs={np.max(np.abs(td)):.3e}")

        fd = ifo.strain_data.frequency_domain_strain
        scenario.logger.info(f"{ifo.name}: fd finite={np.isfinite(fd).all()}, std={np.std(fd):.3e}")

        psd = ifo.power_spectral_density.psd_array
        scenario.logger.info(f"{ifo.name}: psd finite={np.isfinite(psd).all()}, min={np.min(psd):.3e}, max={np.max(psd):.3e}")

    return scenario,logger,plot_dir,out_dir