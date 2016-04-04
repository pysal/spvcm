import numpy as np
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot
from pysal.spreg.diagnostics import constant_check
from warnings import warn as Warn

def weights(W,M, transform):
    try:
        W.transform = transform
        M.transform = transform
    except AttributeError:
        Warn("Weights objects do not support transformation. Proceeding without transforming weights.", UserWarning)
    return W, M

def covariates(X,Z,W,M):
    if Z is None:
        Warn('No upper-level covariates supplied. Setting default to upper-level fixed effects', UserWarning)
        Z = np.ones((M.n, 1))
    else:
        if constant_check(Z):
            raise UserWarning("Z array cannot contain a constant vector; constant will be added automatically")
        else:
            Z = sphstack(np.ones((M.n, 1)), Z) 
    if constant_check(X):
        raise UserWarning("X array cannot contain a constant vector; constant will be added automatically")
    else:
        X = sphstack(np.ones((W.n, 1)), X)
    J, q = Z.shape
    n, p = X.shape
    if J == W.n:
        w = "Z was provided as {} by {}, projecting down to {} by {}"
        Warn(w.format(J, q, n, q))
        Z = spdot(Delta, Z)
    return X,Z

def Delta_members(Delta, membership, N, J):
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

def parameters(rangetup, gridfile, Wmatrix):
    if len(rangetup) == 0 and gridfile is not '':
        promise = lambda : np.load(gridfile) #use closure to lazy compute 
    elif len(rangetup) == 3:
        Warn("Computing grid of log determinants on demand. This may take a while")
        promise = lambda : grid_det(Wmatrix, *rangetup) #again, promise to Base_HSAR
    else:
        raise UserWarning("Length of parameter slice incorrect while no grid file is provided. A range tuple must be (minimum, maximum, step).")
    return promise 


