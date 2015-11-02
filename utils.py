from scipy import sparse as spar
import numpy as np
from numpy import linalg as la
from scipy.sparse import linalg as spla

def logdet(M):
    if spar.issparse(M):
        LU = spla.splu(M)
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        ldet = la.slogdet(M)
    return ldet
