from abc import ABC, abstractmethod
from scenario import GWScenario
from methods import DynestyConfig,PymcNutsConfig,RunMode
from pathlib import Path
import json
from .Pipeline_type import Pipeline_type
import pandas as pd


class Pipeline(ABC):
    def __init__(self,logger,scenario:GWScenario):
        self.logger        = logger
        self.scenario      = scenario
        self.config        = self.getConfig()  #decrepit code, remove it
        self.pipeline_type = None
        
    @abstractmethod
    def getConfig(self):
        raise NotImplementedError
        
    @abstractmethod
    def run(self,runMode:RunMode):
        raise NotImplementedError
    
    def writeMethodResult(self,path,method_meta,waveform_results):
        self.logger.info("$$$ Store data")
        path          = Path(path) / "results.hdf5"
        meta_key      = f"/methods/{self.pipeline_type.code}/meta"
        
        path.parent.mkdir(parents=True, exist_ok=True)

        with pd.HDFStore(path, mode="a", complevel=9, complib="blosc:zstd") as store:
            #create emtpy dataframe so we can attributes at the method level and not the waveform level
            if meta_key not in store:
                store.put(meta_key, pd.DataFrame([{}]), format="fixed")
            store.get_storer(meta_key).attrs.meta_json = json.dumps(method_meta, default=str)
            
            for wf, df in waveform_results.items():
                post_key            = f"/methods/{self.pipeline_type.code}/posterior/{wf}"
                if self.pipeline_type == Pipeline_type.JOINT:
                    posterior = df
                else:
                    posterior = df.posterior
                store.put(post_key, posterior, format="table", data_columns=True)
                st                  = store.get_storer(post_key).attrs
                st.waveform         = wf
                st.n_samples        = int(len(posterior))
                if self.pipeline_type == Pipeline_type.JOINT:
                    info_gain = 0 #0 since the two waveforms ar treated as one
                else:
                    info_gain = df.information_gain
                st.information_gain = info_gain


