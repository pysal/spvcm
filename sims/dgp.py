import numpy as np
from scipy import sparse as spar
from scipy.sparse import linalg as spla
from utils import _mksparse
import pysal as ps
import pandas as pd

def design_matrix(N,J,Delta=None,seed=8879):
    """ 
    Constructs two design matrices, X and Z, 
    
    Parameters
    ===========
    N   :   size of lower-level weights
    J   :   size of upper-level weights
    
    Returns
    =======
    X   :   lower-level design matrix with 3 covariates and a constant
    Z   :   upper-level design matrix with 1 covariate and a constant
    
    Note
    =====
    The size of X and Z are asserted against the input N and J, to ensure no
    mistakes in dimensions have occurred.
    """
    if Delta is None:
        deltawasnone = True
        Delta = _mkDelta(N,J)
    else:
        deltawasnone = False

    np.random.seed(seed=seed)

    x0 = np.ones(N) 
    x1 = np.random.random(size=N)*10 - 5 #continuous x \in [-10,10]
    x2 = np.random.random(size=N)*6 - 3 #continuous x \in [-3,3] won't be significant
    x3 = np.random.randint(3, size=N) - 1 #balanced categorical x \in {-1,0,1}
    X = np.vstack((x0,x1,x2,x3)).T
    N2,p = X.shape #3 covariates + constant
    
    z0 = np.ones(J) 
    z1 = np.random.random(size=J)*8 - 4 #continuous x \in [-8,8]
    Z = np.vstack((z0,z1)).T
    J2, q = Z.shape

    assert N == N2
    assert J == J2
    
    Z = np.dot(Delta, Z)
    
    if deltawasnone:
        return X,Z, Delta
    else:
        return X,Z

def outcome(X, Z, W, M, Delta, rho, lam, 
            Betas=None, Gammas=None, epsilons=None, etas=None, seed=8879):
    """
    Construct the outcome variable given a set of parameters, according to the
    equation

    y = (I - \rho W)^-1(X\beta + Z\gamma + \Delta\theta + (I - \lambda M)\eta +
    \epsilon

    For default, the parameter values are:

    etas ~ N(0,.5)
    epsilons ~ N(0,.7)
    Betas = -3, 6, 0, 2
    Gammas = 5, -4
    """
    np.random.seed(seed=seed)
    N = W.shape[0]
    J = M.shape[0]
    W, M = _mksparse(W, M)
    if etas is None:
        etas = np.random.normal(0,.5, size=J).reshape(J,1)
    if epsilons is None:
        epsilons = np.random.normal(0,.7,size=N).reshape(N,1)
    if Betas is None:
        Betas = np.array([[-3.,6.,0.,2.]]).T
    if Gammas is None:
        Gammas = np.array([[5.,-4.]]).T

    In = spar.identity(N) #will clobber Ipython history
    Ij = spar.identity(J) 
    LU_lo = spla.splu(In - rho * W)
    LU_up = spla.splu(Ij - lam * M)
    Li_lo = LU_lo.solve(In.toarray()) #solves (I - rho W)*X=I, the inverse laplacian
    Li_up = LU_up.solve(Ij.toarray()) #solves (I - lam M)*X=I, ditto above
    covars = np.dot(X, Betas) + np.dot(Z, Gammas)

    inner = covars + np.dot(np.dot(Delta, Li_up), etas) + epsilons
    return np.dot(Li_lo, inner)

def test_space(W,M,**kwargs):
    """
    Construct a testing dataframe containing 1 design matrix (X & Z columns) and
    many outcome vectors (Y columns) over a grid of spatial parameters rho,
    lambda. 

    Parameters
    ===========
    W   :   Lower level pysal weights object
    M   :   Upper level pysal weights object
    
    Optional
    --------
    seed    :   seed to set for random effects generation
    minrho  :   minimum rho value for test grid, minimum for np.arange
    maxrho  :   maximum rho value for test grid, maximum for np.arange
    rhostep :   steps between rho values, stepsize for np.arange
    minlam  :   minimum lambda value for test grid, minimum for np.arange
    maxlam  :   maximum lambda value for test grid, maximum for np.arange
    lamstep :   steps between lambda values, stepsize for np.arange
    Betas   :   vector of lower-level effect sizes (default: .72, 1.3, 0, 2.2)
    Gammas  :   vector of upper-level effect sizes (default: .22, 3.0)
    epsilons:   vector of lower-level random errors (default: normal(0,1,size=N))
    etas    :   vector of upper-level random errors (default: norma(0,.5,size=J))

    Returns
    ========
    fulldata    :   Pandas Dataframe containing design matrix (X, Z columns) and
                    outcome vectors (Y columns) that are suffixed with the rho/lambda 
                    values used. 
    """
    seed = kwargs.pop('seed', 8879)
    minrho = kwargs.pop('minrho', -.8)
    maxrho = kwargs.pop('maxrho', .81) #offset to hit .8 at default step
    rhostep = kwargs.pop('rhostep', .2)
    rhospace = np.arange(minrho, maxrho, rhostep)

    minlam = kwargs.pop('minlam', minrho) #keep trial matrix square
    maxlam = kwargs.pop('maxlam', maxrho)
    lamstep = kwargs.pop('lamstep', rhostep)
    lamspace = np.arange(minlam, maxlam, lamstep)

    N,J = W.n, M.n

    Ws, Ms = _mksparse(W,M, spfunc=lambda x: x.sparse) #Sorry, I'm functional :)

    configurations = [(r,l) for r in rhospace for l in lamspace] #left slower?

    X,Z,Delta = design_matrix(N,J)
    
    p = X.shape[-1]
    q = Z.shape[-1]
    i_in_j = Delta.sum(axis=0)

    dcols = ['X{}'.format(i) for i in range(p)] + ['Z{}'.format(j) for j in range(q)]
    data = pd.DataFrame(np.hstack((X,Z)), columns=dcols)
    
    ycols = ['Y_{}_{}'.format(r,l) for r,l in configurations]
    ydata = np.hstack([outcome(X,Z,Ws,Ms,Delta,r,l,**kwargs) for r,l in configurations])
    ydf = pd.DataFrame(ydata, columns=ycols)

    fulldata = pd.merge(ydf, data, left_index=True, right_index=True)
    return fulldata

def scenario(up, low, **kwargs):
    """
    Construct one square Monte Carlo testing scenario with lower-level
    dimensions low X low and upper-level dimensions up X up.

    Parameters
    ===========
    up  :   side length of upper-level regular lattice
    low :   side length of lower-level regular lattice

    Returns
    ========
    a test space dataframe containing multiple outcome columns constructed over
    the range of parameter values. 
    """
    if type(up) == tuple:
        W_upper = ps.lat2W(*up)
    else:
        W_upper = ps.lat2W(up,up)

    if type(low) == tuple:
        W_lower = ps.lat2W(*low)
    else:
        W_lower = ps.lat2W(low,low)

    W_upper.transform = 'r'
    W_lower.transform = 'r'
    return test_space(W_lower, W_upper, **kwargs), W_upper, W_lower

def _mkDelta(N,J):
    """
    Make an N X J "Delta" matrix recording an individual's membership in a
    group. 
    """
    outmat = np.zeros((N,J))

    splits = np.split(np.arange(0,N), J)
    for col, split in enumerate(splits): 
        subspace = np.ix_(split, np.array([col]))
        outmat[subspace] = 1
    assert outmat.shape == (N,J)
    np.testing.assert_allclose(outmat.sum(axis=1), np.ones(N))
    np.testing.assert_allclose(outmat.sum(axis=0), np.ones(J)*N/J)
    return outmat
