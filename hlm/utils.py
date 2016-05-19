from scipy import sparse as spar
import numpy as np
from numpy import linalg as nla
from scipy.sparse import linalg as spla
from pysal.spreg.opt import requires, simport
from six import iteritems as diter
from functools import wraps
import time
from warnings import warn as Warn
__all__ = ['grid_det']
PUBLIC_DICT_ATTS = [k for k in dir(dict) if not k.startswith('_')]


class Namespace(dict):
    """
    This is a proxy class to add stuff to help with composition. It will expose
    dictionary methods directly to the class's __dict__, meaning it will work
    like a dot-access dictionary. 
    """
    def __init__(self, **kwargs):
        collisions = [k in PUBLIC_DICT_ATTS for k in kwargs.keys()]
        collisions = [k for k,collide in zip(kwargs.keys(), collisions) if collide]
        if len(collisions) > 0:
            raise TypeError('Passing {} to Namespace will overwrite builtin dict methods. Bailing...'.format(collisions))
        self.__dict__.update(kwargs)
        self._dictify()

    def _dictify(self):
        """
        hack to make Namespace pass as if it were a dict by passing methods
        straight through to its own dict
        """
        for method in PUBLIC_DICT_ATTS:
            if method is 'clear':
                continue #don't want to break the namespace
            self.__dict__[method] = eval('self.__dict__.{}'.format(method))

    @property
    def _data(self):
        return {k:v for k,v in diter(self.__dict__) if k not in PUBLIC_DICT_ATTS}

    def __repr__(self):
        innards = ', '.join(['{}:{}'.format(k,v) for k,v in diter(self._data)])
        return '{%s}' % innards
    
    def __getitem__(self, val):
        """
        passthrough to self.__dict__[val]
        """
        return self.__dict__[val]
    
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    
    def __delitem__(self, key):
        del self.__dict__[key]
        _dictify()

    def clear(self):
        not_builtins = {k for k in self.keys() if k not in PUBLIC_DICT_ATTS}
        for key in not_builtins:
            del self.__dict__[key]

def grid_det(W, parmin=-.99, parmax=.99,parstep=.001, grid=None):
    """
    This is a utility function to set up the grid of matrix determinants over a
    range of spatial parameters for a given W. 
    """
    if grid is None:
        grid = np.arange(parmin, parmax, parstep)
    logdets = [splogdet(speye_like(W) - rho * W) for rho in grid]
    grid = np.vstack((grid, np.array(logdets).reshape(grid.shape)))
    return grid

####################
# MATRIX UTILITIES #
####################

def splogdet(M):
    """
    compute the log determinant via an appropriate method. 
    """
    redo = False
    if spar.issparse(M):
        LU = spla.splu(M)
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        sgn, ldet = nla.slogdet(M)
        if np.isinf(ldet) or sgn is 0:
            Warn('Dense log determinant via numpy.linalg.slogdet() failed!')
            redo = True
        if sgn not in [-1,1]:
            Warn("Drastic loss of precision in numpy.linalg.slogdet()!")
            redo = True
        ldet = sgn*ldet
    if redo:
        Warn("Please pass convert to a sparse weights matrix. Trying sparse determinant...", UserWarning)
        ldet = splogdet(spar.csc_matrix(M))
    return ldet

def speye(i, sparse=True):
    """
    constructs a square identity matrix according to i, either sparse or dense
    """
    if sparse:
        return spar.identity(i)
    else:
        return np.identity(i)

spidentity = speye

def speye_like(matrix):
    """
    constructs an identity matrix depending on the input dimension and type
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise UserWarning("Matrix is not square")
    else:
        return speye(matrix.shape[0], sparse=spar.issparse(matrix))

spidentity_like = speye_like

def speigen_range(matrix, retry=True):
    """
    Construct the eigenrange of a potentially sparse matrix. 
    """
    if spar.issparse(matrix):
        emax = spla.eigs(matrix, k=1, which='LR')
        emin = spla.eigs(matrix, k=1, which='SR')
    else:
        try:
            eigs = nla.eigvals(matrix)
            emin, emax = float(eigs.min()), float(eigs.max())
        except Exception as e:
            Warn('Dense eigenvector computation failed!')
            if retry:
                Warn('Retrying with sparse matrix...')
                spmatrix = spar.csc_matrix(matrix)
                speigen_range(spmatrix)
            else:
                Warn('Bailing...')
                raise e 
    return emin, emax
