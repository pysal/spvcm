from scipy import sparse as spar
import numpy as np
from numpy import linalg as nla
from scipy.sparse import linalg as spla
from six import iteritems as diter
from warnings import warn as Warn
import pandas as pd
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

def trace_to_df(trace):
    df = pd.DataFrame().from_records(trace._data)
    for col in df.columns:
        if isinstance(df[col][0], np.ndarray):
            # a flat nested (n,) of (u,) elements hstacks to (u,n)
            new = np.hstack(df[col].values)

            if new.shape[0] is 1:
                newcols = [col]
            else:
                newcols = [col + '_' + str(i) for i in range(new.shape[0])]
            # a df is (n,u), so transpose and DataFrame
            new = pd.DataFrame(new.T, columns=newcols)
            df.drop(col, axis=1, inplace=True)
            df = pd.concat((df[:], new[:]), axis=1)
    return df

####################
# MATRIX UTILITIES #
####################

def splogdet(matrix):
    """
    compute the log determinant via an appropriate method. 
    """
    redo = False
    if spar.issparse(matrix):
        LU = spla.splu(matrix)
        ldet = np.sum(np.log(np.abs(LU.U.diagonal())))
    else:
        sgn, ldet = nla.slogdet(matrix)
        if np.isinf(ldet) or sgn is 0:
            Warn('Dense log determinant via numpy.linalg.slogdet() failed!')
            redo = True
        if sgn not in [-1,1]:
            Warn("Drastic loss of precision in numpy.linalg.slogdet()!")
            redo = True
        ldet = sgn*ldet
    if redo:
        Warn("Please pass convert to a sparse weights matrix. Trying sparse determinant...", UserWarning)
        ldet = splogdet(spar.csc_matrix(matrix))
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

def speigen_range(matrix, retry=True, coerce=True):
    """
    Construct the eigenrange of a potentially sparse matrix. 
    """
    if spar.issparse(matrix):
        try:
            emax = spla.eigs(matrix, k=1, which='LR')[0]
        except (spla.ArpackNoConvergence, spla.ArpackError) as e:
            rowsums = np.unique(np.asarray(matrix.sum(axis=1)).flatten())
            if np.allclose(rowsums, np.ones_like(rowsums)):
                emax = np.array([1])
            else:
                Warn('Maximal eigenvalue computation failed to converge'
                     ' and matrix is not row-standardized.')
                raise e
        emin = spla.eigs(matrix, k=1, which='SR')[0]
        if coerce:
            emax = emax.astype(float)
            emin = emin.astype(float)
    else:
        try:
            eigs = nla.eigvals(matrix)
            emin, emax = eigs.min().astype(float), eigs.max().astype(float)
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
