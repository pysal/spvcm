import numpy as np
from scipy import sparse as spar
from scipy.sparse import linalg as spla
import pysal as ps

def design_matrix(N,J,seed=8879):
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
    np.random.seed(seed=seed)

    x0_CONST = np.random.random()*20 - 10
    x0 = np.ones(N) * x0_CONST #constant

    x1 = np.random.random(size=N)*10 - 5 #continuous x \in [-10,10]

    x2 = np.random.random(size=N)*6 - 3 #continuous x \in [-3,3] won't be significant

    x3 = np.random.randint(3, size=N) - 1 #balanced categorical x \in {-1,0,1}

    X = np.vstack((x0,x1,x2,x3)).T
    N2,p = X.shape #3 covariates + constant
    
    z0_CONST = np.random.random()*2 - 1
    z0 = np.ones(J) * z0_CONST
    z1 = np.random.random(size=J)*8 - 4 #continuous x \in [-8,8]

    Z = np.vstack((z0,z1)).T
    J2, q = Z.shape

    assert N == N2
    assert J == J2

    return X,Z

def outcome(X, Z, W, M, Delta, rho, lam, 
            Betas=None, Gammas=None, epsilons=None, etas=None, seed=8879):
    """
    Construct the outcome variable given a set of parameters, according to the
    equation

    y = (I - \rho W)^-1(X\beta + Z\gamma + \Delta\theta + (I - \lambda M)\eta +
    \epsilon
    """
    np.random.seed(seed=seed)
    N = W.shape[0]
    J = M.shape[0]
    W, M = _mksparse(W, M)
    if etas is None:
        etas = np.random.normal(0,.5, size=M.shape[0]).reshape(J,1)
    if epsilons is None:
        epsilons = np.random.normal(0,1,size=W.shape[0]).reshape(N,1)
    if Betas is None:
        Betas = np.array([[1.,1.,0.,2.]]).T
    if Gammas is None:
        Gammas = np.array([[1,3]]).T

    In = spar.identity(N) #will clobber Ipython history
    Ij = spar.identity(J) 
    LU_lo = spla.splu(In - rho * W)
    LU_up = spla.splu(Ij - lam * M)
    Li_lo = LU_lo.solve(In.toarray()) #solves (I - rho W)*X=I, the inverse laplacian
    Li_up = LU_up.solve(Ij.toarray()) #solves (I - lam M)*X=I, ditto above
    covars = np.dot(X, Betas) + np.dot(Delta, np.dot(Z, Gammas))

    inner = covars + np.dot(np.dot(Delta, Li_up), etas) + epsilons
    return np.dot(Li_lo, inner)

def _mksparse(*args, **kwargs):
    args = list(args)
    spfunc = kwargs.pop('spfunc', spar.csc_matrix)
    for i,arg in enumerate(args):
        if not spar.issparse(arg):
            args[i] = spfunc(arg)
    return args

def _mkDelta(N,J):
    outmat = np.zeros((N,J))

    splits = np.split(np.arange(0,N), J)
    for col, split in enumerate(splits): 
        subspace = np.ix_(split, np.array([col]))
        outmat[subspace] = 1
    assert outmat.shape == (N,J)
    np.testing.assert_allclose(outmat.sum(axis=1), np.ones(N))
    np.testing.assert_allclose(outmat.sum(axis=0), np.ones(J)*N/J)
    return outmat
