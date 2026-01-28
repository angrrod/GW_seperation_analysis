
import json
import corner
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde,entropy
import bilby
import os
from scenario import GWScenario
from pathlib import Path

### helper functions ###
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

def exctractResults(h5_path,logger):
    logger.info("$$$ extract data")
    results   = {}
    with pd.HDFStore(h5_path, "r") as store:
        keys = store.keys()
        methods  = sorted({k.split("/")[2] for k in keys if k.startswith("/methods/")})
        
        for method_name in methods:
            posteriors, diagnostics = readMethodPosteriorInfo(store, method_name,logger)
            methodKey               = f"/methods/{method_name}/meta"
            methodInfo              = readMetaData(store,methodKey,'meta_json',logger)
            methodDict              = {'methodInfo' : methodInfo}
            results[method_name]    =  {'diagnostics' : methodDict | diagnostics,'posteriors': posteriors}
    return results
    
def readMethodPosteriorInfo(store, method_name,logger):
    logger.info("$$$ read method info")
    posteriors  = {}
    diagnostics = {}
    for wf in ("waveFormA", "waveFormB"):
        key = f"/methods/{method_name}/posterior/{wf}"
        if key in store:
            waveformDict = {}
            posteriors[wf]  = store[key]
            waveformDict['information_gain'] = readMetaData(store,key,'information_gain',logger)
            diagnostics[wf] = waveformDict
    return posteriors, diagnostics

def readMetaData(store,key,attribute,logger):
    logger.info("$$$ read meta data")
    attrs = store.get_storer(key).attrs
    raw = getattr(attrs, attribute) 

    # 1) Numeric attributes: already a number → return as Python scalar
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return raw.item() if hasattr(raw, "item") else raw
    
    # 2) Bytes-like: decode to str
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        raw = raw.decode("utf-8")
        
    # 3) NumPy string-like: convert to Python str
    if isinstance(raw, np.str_):
        raw = str(raw)
        
    # 4) If it's a normal string, optionally parse JSON if it looks like JSON
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") or s.startswith("["):
            return json.loads(s)   # JSON payload
        return raw                # plain string payload
    
    raise TypeError(
        f"Unsupported attribute type {type(raw)} for {key}/{attribute}"
    )

def calculate_KL_between_joint(results,method_code:str,logger):
    logger.info("$$$ getting best KL divergence")
    jointRes   = results['joint_likl']['posteriors']
    methodRes  = results[method_code]['posteriors']
    meanKL     = []
    meanKL_rev = []
    combos     = _get_combinations(jointRes,methodRes,logger)
    for combo in combos:
        KL_div     = []  #joint is first -> we max over joint since joint is the default
        KL_div_rev = []  #method is first
        for KL_calc in combo:
            KL_div.append(_kl_divergence_kde(KL_calc[0],KL_calc[1],logger))
            KL_div_rev.append(_kl_divergence_kde(KL_calc[1],KL_calc[0],logger))
        meanKL.append(np.mean(KL_div))
        meanKL_rev.append(np.mean(KL_div_rev))
    KL_res     = min(meanKL)
    KL_res_rev = meanKL_rev[meanKL.index(KL_res)]
    return KL_res,KL_res_rev

def _kl_divergence_kde(p_samples, q_samples,logger):
    logger.info("$$$ getting KL divergence between 2 posteriors using KDE")
    p_samples = p_samples.copy()
    q_samples = q_samples.copy()
    
    rel_cols  = p_samples.columns.intersection(q_samples.columns)
    q_samples = q_samples[rel_cols]
    p_samples = p_samples[rel_cols]
    
    kde_p = gaussian_kde(p_samples.T)
    kde_q = gaussian_kde(q_samples.T)

    combined_sample = pd.concat([p_samples, q_samples], axis=0)
    return entropy(kde_p.pdf(combined_sample.T),kde_q.pdf(combined_sample.T))

def _get_combinations(jointRes:dict,methodRes:dict,logger):
    logger.info("$$$ getting all possible waveform combinations")
    #TODO: extend to more waves if computationaly feasable
    if len(methodRes.keys()) == 1:
        return [
            [
                (jointRes.get('waveFormA'),methodRes.get('waveFormA'))
            ],
            [
                (jointRes.get('waveFormB'),methodRes.get('waveFormA'))
            ]
            ]
    else:
        return [[
            (jointRes.get('waveFormA'),methodRes.get('waveFormA')),
            (jointRes.get('waveFormB'),methodRes.get('waveFormB'))
            ],
            [
            (jointRes.get('waveFormA'),methodRes.get('waveFormB')),
            (jointRes.get('waveFormB'),methodRes.get('waveFormA'))
            ],
                ]

def getMaximumLikelihood(result):
    posterior = result.posterior
    idx_ml    = posterior["log_likelihood"].idxmax()
    ml_sample = posterior.loc[idx_ml]
    return ml_sample 

def setUpLoggerScenario(ScenConfig):
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
    
    scenario = GWScenario(logger, ScenConfig)
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

def get_base_log_dir() -> Path:
    return get_base_work_dir() / "logs"

def get_postprocessing_dir() -> Path:
    return get_base_work_dir() / "postProcessing"

def get_base_work_dir() -> Path:
    """
    Root directory for all generated artifacts (logs, postProcessing, results, …).
    """
    if "VSC_DATA" in os.environ:
        return Path(os.environ.get("GW_INP_DIR", os.environ["VSC_DATA"]))
    return Path(".")