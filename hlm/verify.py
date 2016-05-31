import numpy as np
from pysal.spreg.utils import sphstack 
from pysal.spreg.diagnostics import constant_check
from warnings import warn as Warn

def weights(W,M, transform):
    """
    This tries to transform a pysal spatial weights object into being
    row-standardized. It warns if the objects do not support transformation. 

    """
    try:
        W.transform = transform
        M.transform = transform
    except AttributeError:
        Warn("Weights objects do not support transformation. Proceeding without transforming weights.", UserWarning)
    return W, M

def covariates(X,W):
    """
    This 

    1. checks if the lower-level covariate contains a constant
    2. adds a constant to the lower-level covariate if it has no constant
    """
    if constant_check(X):
        Warn("X array should not contain a constant vector;"
             " constant will be added automatically")
    else:
        X = sphstack(np.ones((W.n, 1)), X)
    
    return X

def Delta_members(Delta, membership, N, J):
    """
    This computes and verifies a Delta or membership vector. 
    """
    if Delta is None and membership is None:
        raise UserWarning("No Delta matrix nor membership classification provided. Refusing to arbitrarily assign units to upper-level regions.")
    elif membership is None:
        membership = np.zeros((J,1))
        for idx, region in enumerate(Delta.T):
            membership[region.flatten() == 1] = idx
    elif Delta is None:
        Delta = np.zeros((N, J))
        for idx, region in enumerate(np.unique(membership)):
            Delta[membership.flatten() == region, idx] = 1
    else:
        raise UserWarning("Both Delta and Membership vector provided. Please pass only one or the other.")
    return Delta, membership
