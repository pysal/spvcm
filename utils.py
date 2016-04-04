from scipy import sparse as spar
import numpy as np
from numpy import linalg as nla
from scipy.sparse import linalg as spla
from pysal.spreg.opt import requires, simport
import time
__all__ = ['grid_det']

def splogdet(M):
    """
    compute the log determinant via an appropriate method. 
    """
    if spar.issparse(M):
        LU = spla.splu(M)
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        ldet = la.slogdet(M)
    return ldet

def speye(i, sparse=True):
    """
    constructs a square identity matrix according to i, either sparse or dense
    """
    if sparse:
        return SP.identity(i)
    else:
        return np.identity(i)

def speye_like(matrix):
    """
    constructs an identity matrix depending on the input dimension and type
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise UserWarning("Matrix is not square")
    else:
        return speye(matrix.shape[0], sparse=SP.issparse(matrix))

def inversion_sample(pdvec, grid):
    """
    sample from a probability distribution vector, according to a grid of values
    """
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    np.testing.assert_allclose(np.cumsum[-1], 1)
    while True:
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

def grid_det(W, emin=-.99, emax=.99,step=.001):
    """
    This is a utility function to set up the grid of matrix determinants over a
    range of spatial parameters for a given W. 
    """
    grid = np.arange(emin, emax, step)
    logdets = [splogdet(speye_like(W) - rho * W) for rho in grid]
    grid = np.vstack((grid, np.array(logdets).reshape(grid.shape)))
    return grid

_, T = simport('theano.tensor')
_, th = simport('theano')
_, tla = simport('theano.tensor.nlinalg')

if T is not None:
    W = T.dmatrix('W')
    param = T.dscalar('param')
    I = T.identity_like(W)
    svd = tla.svd(I - param * W)
    det = T.abs_(svd[1]).log().sum()
    
    _theano_det = th.function([W, param], det, allow_input_downcast=True) 

    def theano_grid_det(W, emin=-.99, emax=.99, step=.001):
        """
        This is a theano version of the gridded determinant function
        """
        grid = np.arange(emin, emax, step)
        logdets = [_theano_det(W, par) for par in grid]
        return np.vstack((grid, np.array(logdets).reshape(grid.shape)))
