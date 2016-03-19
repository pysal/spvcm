from scipy import sparse as spar
import numpy as np
from numpy import linalg as la
from scipy.sparse import linalg as spla
import time

def logdet(M):
    if spar.issparse(M):
        LU = spla.splu(M)
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        ldet = la.slogdet(M)
    return ldet

def time_gibbs(Gibbs, n=100):
    times = {k:[] for k in Gibbs.var_names}
    for it in range(n*len(Gibbs.var_names)):
        currname = Gibbs.var_names[it % len(Gibbs.var_names)]
        s = time.time()
        next(Gibbs)
        times[currname].append(time.time() - s)
    return times

def inversion_sample(pdvec, grid=None):
    if not np.allclose(pdvec.sum(), 1):
        pdvec = pdvec / pdvec.sum()
    cdvec = np.cumsum(pdvec)
    np.testing.assert_allclose(np.cumsum[-1], 1)
    while True:
        rval = np.random.random()
        topidx = np.sum(cdvec <= rval) -1
        if topidx >= 0:
            return grid[topidx]

