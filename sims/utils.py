from scipy import sparse as spar
from scipy.sparse import linalg as spla
import numpy as np

def logdet(M, LU = None):
    """
    Compute the log determinant of a sparse matrix 
    using a numerically-stable sparse method relying on LU decomposition.
    """
    if LU is None:
        M = _mksparse(M) #returns a list always
        LU = spla.splu(M)
    return np.sum(np.log(np.abs(LU.U.diagonal())))

def invert(M, LU=None, spfunc=None):
    """
    Compute the inverse of a matrix using a numerically-stable and memory
    efficient sparse method relying on LU decomposition
    """
    if LU is None:
        M = _mksparse(M)
        LU = spla.splu(M)
        I = np.identity(M.shape[0])
    return LU.solve(I)

def _mksparse(*args, **kwargs):
    """
    sparsify any number of matrices using a sparsification function
    """
    args = list(args)
    spfunc = kwargs.pop('spfunc', spar.csc_matrix)
    for i,arg in enumerate(args):
        if not spar.issparse(arg):
            args[i] = spfunc(arg)
    if len(args) == 1:
        return args[0]
    else:
        return args
