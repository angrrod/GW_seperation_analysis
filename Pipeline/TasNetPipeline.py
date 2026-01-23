from methods import MethodConfig, SingleLikelihoodMethod, RunMode
from .Pipeline_type import Pipeline_type
from scenario import GWScenario
import copy
import numpy as np
import bilby
import torch
from .Pipeline import Pipeline
from bilby.gw.utils import noise_weighted_inner_product

from config.option import parse
import trainer.denoise_pytorch_trainer
from model.model_rnn import Dual_RNN_model
from trainer.end_to_end_denoise_saparate_trainer import CombinedModel

class TasNetPipeline(Pipeline):
    def __init__(self,logger,scenario:GWScenario,config:MethodConfig):
        super().__init__(logger, scenario, config)
        self.pipeline_type    = Pipeline_type.TASNET
        self.singleSampler1 = SingleLikelihoodMethod(scenario, logger,config,self.pipeline_type.code,"_1")
        self.singleSampler2 = SingleLikelihoodMethod(scenario, logger,config,self.pipeline_type.code,"_2")
        self.splitter       = self.getTasNetSplitter() #can be used to analyze other splitters
    
    def run(self,runMode:RunMode):
        #take splitted signal and perform single signal on both residual ifos
        self.logger.info("$$$ generating samples")
        ifos_A,ifos_B = self.getResdiualIfos()
        resultsSampleA = self.singleSampler1.run(runMode,ifos_override = ifos_A)
        resultsSampleB = self.singleSampler2.run(runMode,ifos_override = ifos_B)
        results = {
            "waveFormA" : resultsSampleA,
            "waveFormB" : resultsSampleB
        }
        return results
        
    
    def getResdiualIfos(self):
        self.logger.info("$$$ get residual ifos")
        
        ifos_A = []
        ifos_B = []
        
        for i,ifo in enumerate(self.scenario.ifos):
            self.logger.info(f"$$$ splitting ifo: {i}")
            strain    = torch.tensor(ifo.strain_data.time_domain_strain).unsqueeze(0)
            waveForms = [torch.squeeze(s).detach().cpu().numpy() for s in self.splitter(strain)] 
            # earliest peak is selected according to the observation -> this will not work if the stations are too far and if the overlap of the peak is very close.
            # to be precise the worst case scenario is if the interferromters are opposite on earth and if the sources lie on this line. we assume that one wave hits
            # earth in one detector in the time it travels to the other station through the earths crust, if the first is hit but the second isn't, whilst the other wave hits the second station
            # the order of the waves will be reversed. this leads to a time of 'diam of earth'/'speed of light in vacuum' which gives around
            # 0.04s.
            # if far away satelites are used this effect coud become more significant.
            if np.argmax(waveForms[0]) >= np.argmax(waveForms[1]):
                wave_A_norm = waveForms[0]
                wave_B_norm = waveForms[1]
            else:
                wave_A_norm = waveForms[1]
                wave_B_norm = waveForms[0]
            
            a1,a2 = self.getOptimalAmplitudes(waveForms,ifo)
            self.logger.info(f"$$$ optmial amplitudes {a1} and {a2}")
            
            ifo_A,_ = self.getResidualIfos_time([wave_A_norm],[a1],ifo)
            ifo_B,_ = self.getResidualIfos_time([wave_B_norm],[a2],ifo)
            ifos_A.append(ifo_A) 
            ifos_B.append(ifo_B) 
            
        ifos_A = bilby.gw.detector.InterferometerList(ifos_A)
        ifos_B = bilby.gw.detector.InterferometerList(ifos_B)
        
        return ifos_A,ifos_B
    
    def sampleAmplitudes(self,waveForms,ifo):
        #sample the amplitus for the normalized waveforms
        #set priors to realistic values of 5 sigma away
        priors       = bilby.core.prior.PriorDict()
        sigma        = np.std(ifo.strain_data.time_domain_strain)
        priors["a1"] = bilby.core.prior.Uniform(-5*sigma, 5*sigma, "a1")
        priors["a2"] = bilby.core.prior.Uniform(-5*sigma, 5*sigma, "a2")
        
        class amplitudeLikelihood(bilby.core.likelihood.Likelihood):
            #special likelihood for sampeling of amplitudes
            def __init__(self, waveForms, ifo,parent):
                # declare parameters that will be optimized / sampled
                super().__init__(parameters={"a1": None, "a2": None})
                self.waveForms = np.asarray(waveForms)
                self.ifo       = ifo
                self.parent    = parent

            def log_likelihood(self):
                a1 = self.parameters["a1"]
                a2 = self.parameters["a2"]
                _,likelihood = self.parent.getResidualIfos_time(self.waveForms,[a1,a2],self.ifo)
                return likelihood
            
        result = bilby.run_sampler(
            likelihood = amplitudeLikelihood(waveForms,ifo,self),
            priors     = priors,
            sampler    = "emcee",
            nwalkers   = 32,        # must be >= 2*ndim; ndim=2 here
            nsteps     = 5000,
            burn_in    = 1000,
            thin_by    = 10,
            resume     = False,
            outdir     = "logs/log_ET_dynesty_"+ self.pipeline_type.code+ "_Amplitude",
            label      = "ml_2d",
            )
        return result
    
    def getOptimalAmplitudes(self,waveForms,ifo):
        #we use optimal posterior distribution 
        result = self.sampleAmplitudes(waveForms,ifo)
        i_ml = result.posterior["log_likelihood"].idxmax()
        a1 = float(result.posterior.loc[i_ml, "a1"])
        a2 = float(result.posterior.loc[i_ml, "a2"])
        return a1, a2
    
    def _strip_prefix_if_present(self,state_dict, prefix="module."):
        # Only needed if a checkpoint was saved from nn.DataParallel, needed to obtain correct instances from the end-to-end checkpoint.
        if not any(k.startswith(prefix) for k in state_dict.keys()):
            return state_dict
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    
    def getTasNetSplitter(self):
        # returns the feed forward model that splits the strain into normalized strains.
        self.logger.info("$$$ split signal via TasNet")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) Instantiate architectures (random init is fine; checkpoint will overwrite)
        denoise = trainer.denoise_pytorch_trainer.MyModel().to(device)

        opt = parse("/root/phd/GW_seperation_analysis/External/gravitational-wave-separation/config/Dual_RNN/train_rnn.yml")
        sep = Dual_RNN_model(**opt["Dual_Path_RNN"]).to(device)

        # 2) Combine
        e2e = CombinedModel(denoise, sep).to(device)

        # 3) Load END-TO-END checkpoint 
        ckpt = torch.load("/root/phd/GW_seperation_analysis/External/end_to_end_model/best_end2end.pt", map_location=device)
        state = self._strip_prefix_if_present(ckpt["model_state_dict"])
        missing, unexpected = e2e.load_state_dict(state, strict=False)
        if missing:
            print("Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print("Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

        e2e.eval()
        return e2e
    
    def getResidualIfos_time(self,waveForms,amplitudes,ifo):
        #returns the static likelihood of the residual with amplitudes and waveforms, can be used for model validation and sampling of amplitudes.
        if len(waveForms) != len(amplitudes):
            raise AssertionError(f"both waveForms and amplitudes must have the same lenght but got {len(waveForms)} waveforms and {len(amplitudes)} amplitudes")
        ifo_copy        = copy.deepcopy(ifo)
        residual_strain = ifo_copy.strain_data.time_domain_strain
        for i,waveForm in enumerate(waveForms):
            residual_strain = residual_strain - amplitudes[i]*waveForm
            
        #set the strain to allow the internal workings of bilby to convert to freq domain
        ifo_copy.strain_data.set_from_time_domain_strain(
            time_domain_strain = residual_strain,
            sampling_frequency = ifo.strain_data.sampling_frequency,
            duration           = ifo.strain_data.duration,
            start_time         = ifo.strain_data.start_time,
        )
        
        #we use the discretized version of the inner product integral (for which we need df and mask)
        psd_array   = ifo_copy.power_spectral_density_array
        residua_feq = ifo_copy.strain_data.frequency_domain_strain
        df          = ifo_copy.frequency_array[1] - ifo_copy.frequency_array[0] #get frequency bin size
        mask        = ifo_copy.frequency_mask #get actual used frequencies

        rr = noise_weighted_inner_product(
            residua_feq[mask],
            residua_feq[mask],
            psd_array[mask],
            df,
        )
        logl = -0.5 * rr
        return ifo_copy,float(logl)