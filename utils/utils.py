
import os,json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner
import bilby
from scenario import ScenarioConfig
from scenario import GWScenario
from pathlib import Path
import pandas as pd
import numpy as np

### helper functions ###



def plotOverlap(params,results,truths,logger,plot_dir):
    logger.info("$$$ Making corner plots")
    
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
        res = results[method].get('posteriors')
        for waveform in res:
            wave = res[waveform]
            color = colors[color_idx % n_colors]
            color_idx += 1            #separate the joint poisterior that ends with _A and _B in their respective posterior samples
            label = f"{method} – {waveform}"
            
            # if method == Method_type.JOINT:
            #     df_A = wave.copy()
            #     df_A.rename(columns=params_A, inplace=True)
            #     df_B = wave.copy()
            #     df_B.rename(columns=params_B, inplace=True)
            #     fig = createCornerPlot(df_A[params].values,params,color,fig,truths[1])  #index doesn't matter as things get overlapped
            #     fig = createCornerPlot(df_B[params].values,params,color,fig,truths[0])
            # else:
                #Single waveform
            fig = createCornerPlot(wave[params].values,params,color,fig,truths[0])
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

def setUpLoggerScenario(ScenConfig):
    #set-up plotting dirs
    out_dir = "postProcessing"
    plot_dir = os.path.join(out_dir, "Plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    #Logger
    bilby.core.utils.setup_logger(
        log_level="DEBUG", #INFO
        label="my_run", 
        outdir="logs",
    )
    logger = bilby.core.utils.logger
    logger.info("$$$ start_run")
    
    scenario       = GWScenario(logger, ScenConfig)
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

def writeMethodResult(path,method_name,method_meta,waveform_results,logger):
    logger.info("$$$ Store data")
    path = Path(path) / "results.hdf5"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_key = f"/methods/{method_name}/meta"
    
    with pd.HDFStore(path, mode="a", complevel=9, complib="blosc:zstd") as store:
        #create emtpy dataframe so we can attributes at the method level and not the waveform level
        if meta_key not in store:
            store.put(meta_key, pd.DataFrame([{}]), format="fixed")
        store.get_storer(meta_key).attrs.meta_json = json.dumps(method_meta, default=str)
        
        for wf, df in waveform_results.items():
            post_key            = f"/methods/{method_name}/posterior/{wf}"
            store.put(post_key, df.posterior, format="table", data_columns=True)
            st                  = store.get_storer(post_key).attrs
            st.waveform         = wf
            st.n_samples        = int(len(df.posterior))
            st.information_gain = df.information_gain

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
            posteriors[wf]  = store[key]
            diagnostics['methodInfo'] = readMetaData(store,key,'information_gain',logger)
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
