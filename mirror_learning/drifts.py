import numpy as np
import torch

def Zero():
    return lambda pi1, pi2: torch.zeros(pi1.shape[0])

def TVSq(scale=1.):
    return lambda pi1, pi2: scale*0.5*torch.sum(torch.abs(pi1-pi2), dim=-1)**2

def EuclidSq(scale=1.):
    return lambda pi1, pi2: scale*torch.sum( (pi1-pi2)**2, dim=-1 )

def KL(scale=1.):
    return lambda pi1, pi2: scale*torch.sum(pi1*torch.log((pi1+1e-6)/(pi2+1e-6)), dim=-1)

def expected_drift(mirror, beta, functional=True):
    if functional:
        return lambda Pi1, Pi2: torch.dot(beta(Pi1), mirror(Pi1, Pi2))

    return lambda Pi1, Pi2: torch.dot(beta, mirror(Pi1, Pi2))


def min_drift(drift):
    return lambda Pi1, Pi2: torch.min(drift(Pi1, Pi2))

