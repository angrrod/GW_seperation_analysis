from .Method import Method
from .MethodConfig import MethodConfig
from .Method_type import Method_type
from scenario import GWScenario

# import torch
# from config.option import parse
# import trainer.denoise_pytorch_trainer
# from model.model_rnn import Dual_RNN_model
# from trainer.end_to_end_denoise_saparate_trainer import CombinedModel

# class TasNetMethod(Method):
#     def __init__(self, run_sampler:bool,scenario:GWScenario,logger,config:MethodConfig,start_from_chekpt):
#         super().__init__( run_sampler, scenario, logger, config,start_from_chekpt)
#         self.method_type = Method_type.SINGLE
    
#     def generateSamples(self):
#         #take splitted signal and perform single signal on both residual ifos
#         pass
    
#     def runTasNetModel(self):
#         #result in ML estimates for 
#         pass
        
#     def getSplittedWaveParams(self):
#         #returns the feed forward model that splits the strain into normalized strains.
        
#         device = "cuda" if torch.cuda.is_available() else "cpu"

#         # 1) Denoise model
#         denoise = denoise_pytorch_trainer.MyModel().to(device)
#         ckpt1   = torch.load("./checkpoint_mse/MyModel/best.pt", map_location=device)
#         denoise.load_state_dict(ckpt1["model_state_dict"])
#         denoise.eval()

#         # 2) Separator model (Dual-Path RNN)
#         opt   = parse("./config/Dual_RNN/train_rnn.yml")
#         sep   = Dual_RNN_model(**opt["Dual_Path_RNN"]).to(device)
#         ckpt2 = torch.load("./checkpoint/Dual_Path_RNN/best_rnn.pt", map_location=device)
#         sep.load_state_dict(ckpt2["model_state_dict"])
#         sep.eval()

#         # 3) End-to-end wrapper
#         e2e = CombinedModel(denoise, sep).to(device)
#         e2e.eval()

#         return e2e
    
#     def sampleAmplitudes(self):
#         pass