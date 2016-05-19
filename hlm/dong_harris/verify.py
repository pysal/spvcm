import numpy as np
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot
from ..utils import grid_det
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
        for region in np.unique(membership):
            Delta[membership == region] = 1
    else:
        raise UserWarning("Both Delta and Membership vector provided. Please pass only one or the other.")
    return Delta, membership

def parameters(gridspec, gridfile, Wmatrix):
    """
    This returns a lambda that, when evaluated, either 
    
    1. loads the log determinant grid from gridfile
    2. computes the log determinant for each value of gridspec
    3. computes the log determinant for np.arange(*gridspec)
    """
    if gridspec is None and gridfile is not '':
        promise = lambda : grid_from_file(gridfile) #use closure to lazy load 
    elif isinstance(gridspec, np.ndarray):
        Warn("Computing grid of log determinants on demand may take a while")
        return lambda : grid_det(Wmatrix, grid=gridspec) 
    elif len(gridspec) == 3:
        Warn("Computing grid of log determinants on demand may take a while")
        parmin, parmax, parstep = gridspec
        promise = lambda : grid_det(Wmatrix, parmin=parmin, parmax=parmax,
                                    parstep=parstep) #again, promise to Base_HSAR
    else:
        raise UserWarning("Length of parameter slice incorrect while no grid file is provided. A range tuple must be (minimum, maximum, step).")
    return promise 

def grid_from_file(gridfile):
    """
    read and verify a sampling grid from a numpy file
    """
    data = np.load(gridfile)
    if data.shape[1] == 2:
        data = data.T
    elif data.shape[0] != 2:
        raise UserWarning('Grid read from {} is not correctly formatted. The grid must be a (2,k) array, where k is the number of gridpoints.')
    return data
