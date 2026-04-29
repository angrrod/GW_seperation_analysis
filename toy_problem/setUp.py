import torch
import os

def getModelParams(isIndependentCopula:bool = True):
    #prior and likl models for pyro
    if isIndependentCopula:
        cov = 10*torch.eye(4)
    else:
        cov = 10*torch.tensor([ 
                [1.0, 0.5, 0.1, 0.3],
                [0.5, 1.0, 0.2, 0.05],
                [0.1, 0.2, 1.0, 0.45],
                [0.3, 0.05, 0.45, 1.0],
            ])
        # cov = 10*torch.eye(4)
    params = {
        "means"    : torch.tensor([0.0,0.0,0.0,0.0]),
        "cov"      : cov,
        "priorVar" : 10,
    }
    return params

def getPltDir():
    dir = "toy_problem/plots/"
    os.makedirs(dir, exist_ok=True)
    return dir
