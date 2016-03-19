from scipy import sparse as spar
from scipy.sparse import linalg as spla
import numpy as np

def logdet(M, **kwargs):
    """
    Construct the log determinant of a matrix M using appropriate methods,
    depending on whether M is sparse or dense. 
    """
    if spar.issparse(M):
        return LU_logdet(M, **kwargs)
    else:
        r= np.linalg.slogdet(M)
        try:
            assert np.abs(r[0]) in [0,1]
            return r[0]*r[1]
        except AssertionError:
            return LU_logdet(_mksparse(M), **kwargs)

def LU_logdet(M, LU = None):
    """
    Compute the log determinant of a sparse matrix 
    using a numerically-stable sparse method relying on LU decomposition.
    """
    if LU is None:
        M = _mksparse(M) #returns a list always
        LU = spla.splu(M)
    return np.sum(np.log(np.abs(LU.U.diagonal())))

def invert(M, **kwargs):
    """
    invert an arbitrarily-shaped matrix using either LU methods or using default
    scipy inverse matrix computation. 
    """
    if M.shape[0] != M.shape[1]:
        return spla.inv(M)
    else:
        return LU_invert(M, **kwargs)

def lstsq(M,B, **kwargs):
    """
    Compute the solution to Mx = B for arbitrary M, B
    """
    if M.shape[0] != M.shape[1]:
        return spla.lsqr(M, B, **kwargs)[0]
    else:
        return LU_lstsq(M, B, **kwargs)


def LU_invert(M, LU=None, spfunc=None):
    """
    Compute the inverse of a matrix using a numerically-stable and memory
    efficient sparse method relying on LU decomposition
    """
    I = spar.identity(M.shape[0])
    return LU_lstsq(M, I, LU, spfunc)

def LU_lstsq(M, B, LU=None, spfunc=None):
    if LU is None:
        M = _mksparse(M)
        LU = spla.splu(M)
    return LU.solve(B.toarray())

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

def time_gibbs(Gibbs, n=100):
    """
    time n iterations of gibbs sampler Gibbs
    """
    times = {k:[] for k in Gibbs.var_names}
    for it in range(n*len(Gibbs.var_names)):
        currname = Gibbs.var_names[it % len(Gibbs.var_names)]
        s = time.time()
        next(Gibbs)
        times[currname].append(time.time() - s)
    return times

def inversion_sample(pdvec, grid=None):
    """
    the inversion sampling function required by samplers
    """
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    while True:
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

