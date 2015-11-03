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
        Gibbs.next()
        times[currname].append(time.time() - s)
    return times

