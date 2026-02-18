from .Pipeline import Pipeline
from .Pipeline_type import Pipeline_type
from methods import SingleLikelihoodMethod,DynestyConfig,RunMode
from scenario import GWScenario
from bilby.gw.detector import InterferometerList
import copy
import numpy as np
from utils import getMaximumLikelihood
import os
from utils import get_postprocessing_dir
class HierarchicalPipeline(Pipeline):
    def __init__(self,logger,scenario:GWScenario):
        super().__init__(logger, scenario)
        self.pipeline_type           = Pipeline_type.HIERARCHICAL
        self.singleSampler1          = SingleLikelihoodMethod(scenario, logger,self.config,self.pipeline_type.code,"_1") #run_sampler to true so that we always generate a new sample instead of using the one from the single method
        self.singleSampler2          = SingleLikelihoodMethod(scenario, logger,self.config,self.pipeline_type.code,"_2")
        
    def run(self,runMode:RunMode):
        self.logger.info("$$$ generate Samples for hyrarchical model")
        resultsSampleA        = self.singleSampler1.run(runMode,ifos_override = None)
        MLPosteriorA          = getMaximumLikelihood(resultsSampleA)
        
        #TODO: remove
        # MLPosteriorA['luminosity_distance'] = 100
        # MLPosteriorA['geocent_time']        = 1
        
        residualIfos          = self.getResidualIfos_freq(MLPosteriorA)
        
        for i in range(len(residualIfos)):
            ifo1 = residualIfos[i]
            ifo2 = self.scenario.ifos[i]
            d1 = ifo1.strain_data.frequency_domain_strain
            d2 = ifo2.strain_data.frequency_domain_strain

            self.logger.info(f"$$$ FD shapes:, {d1.shape}, {d2.shape}")

            diff_norm = np.linalg.norm(d1 - d2)
            self.logger.info(f"$$$ ||d1 - d2|| = {diff_norm}")
            diff = np.linalg.norm(d1 - d2)
            n1   = np.linalg.norm(d1)
            n2   = np.linalg.norm(d2)

            self.logger.info(f"$$$ ||d1||, {n1}, ||d2||, {n2}, ||d1-d2||, {diff}, rel, {diff / (n1 + 1e-300)}")
            self.logger.info(f"$$$ max|d1|, {np.max(np.abs(d1))}, max|d2|, {np.max(np.abs(d2))}, max|diff|, {np.max(np.abs(d1-d2))}")
            
        #test residual ifo
        plot_dir = os.path.join(get_postprocessing_dir(), "Plots")
        self.scenario.makePlots(["strain_time_domain_set_up_hyrarchical","qtransform_set_up_hyrarchical"],plot_dir,ifos = residualIfos)
        
        resultsSampleB        = self.singleSampler2.run(runMode,ifos_override = residualIfos) #if we marginalize, the resampeling of the general posterior has been done via generateSamples() in the super class. 
        results = {
            "waveFormA" : resultsSampleA,
            "waveFormB" : resultsSampleB
        }
        return results
        
    def getResidualIfos_freq(self,MLPosteriorA):
        """returns the residual ifos without the MLPosterior waveform, 
        used for residual analysis of the method.
        Uses the frequency domain"""
        new_ifos     = []
        pols    = self.scenario.wg.frequency_domain_strain(parameters=dict(MLPosteriorA)) #returns cross and plus waveform
        for ifo in self.scenario.ifos:
            # get polarizations
            h_fd    = ifo.get_detector_response(pols, dict(MLPosteriorA))
            d_fd    = ifo.strain_data.frequency_domain_strain
            res_fd  = d_fd - h_fd  #TODO fix
            new_ifo = self.clone_ifo_with_new_fd_strain(ifo,res_fd)
            new_ifos.append(new_ifo)
        return InterferometerList(new_ifos)
    
    def clone_ifo_with_new_fd_strain(self, ifo, new_fd):
        """
        Diagnostic: return an IFO that is identical to `ifo` in every way,
        except that its frequency_domain_strain is replaced by `new_fd`.

        This avoids losing geometry/calibration/min-max-freq/windowing metadata
        that you would lose with get_empty_interferometer().
        """
        if new_fd.shape != ifo.strain_data.frequency_domain_strain.shape:
            raise ValueError(
                f"FD strain shape mismatch for {ifo.name}: "
                f"new_fd {new_fd.shape} vs original {ifo.strain_data.frequency_domain_strain.shape}"
            )
            
        new_ifo = copy.deepcopy(ifo)
        new_ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain = np.array(new_fd, copy=True),
            sampling_frequency      = ifo.strain_data.sampling_frequency,
            duration                = ifo.strain_data.duration,
            start_time              = ifo.strain_data.start_time,
        )
        return new_ifo
        
    def getConfig(self):
        return DynestyConfig()
