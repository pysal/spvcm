import numpy as np
import scipy.linalg as scla
import pandas as pd
from hlm.dong_harris.verify import Delta_members
import pysal as ps
from warnings import warn as Warn

np.random.seed(8879)

def dgp(Betas, Sigma2_e, Sigma2_u, Rho, W, Lambda, M, Delta):
    """
    Compute one outcome vector from one input vector.

    Arguments
    -----------
    Betas
    Sigma2_e
    Sigma2_u
    Rho
    W
    Lambda
    M
    Delta
    """
    W.transform = M.transform = 'r'
    Wmat = W.sparse
    Mmat = M.sparse
    In = np.identity(W.n) 
    Ij = np.identity(M.n)

    X = np.random.normal(0, 10, size=(W.n, Betas.shape[0]))
    u = np.random.normal(0, Sigma2_u, size=(M.n, 1))
    Thetas = np.dot(np.asarray(scla.inv(Ij - Lambda * Mmat)), u)
    e = np.random.normal(0, Sigma2_e, size=(W.n,1))
    kernel = 4 + np.dot(X, Betas) + np.dot(Delta, Thetas) +e
    A = np.asarray(scla.inv(In - Rho * Wmat))
    y = np.dot(A, kernel)
    return y, X

def scenario_square(N,J, **kw):
    """
    build a square testing scenario for the HSAR
    """
    sqrtN = np.sqrt(N)
    sqrtJ = np.sqrt(J)
    W = ps.lat2W(sqrtN, sqrtN)
    M = ps.lat2W(sqrtJ, sqrtJ)
    W.transform = M.transform = 'r'
    N_per_J = N / float(J)
    membership = list(range(0,J-1)) * N_per_J
    return scenario_from_membership(membership, W, M, **kw)

def minnesota(**kw):
    """
    Build a HSAR setup from the minnesota radon lattices, with county-level
    spatially-autocorrelated random effects and household-level observations
    """
    W = ps.open('./test_data/w_lower.mtx').read()
    M = ps.open('./test_data/w_upper.mtx').read()
    W.transform = M.transform = 'r'
    df = pd.read_csv('./test_data/test.csv')
    membership = df[['county']].values -1
    return scenario_from_membership(membership, W, M,  **kw)

def south_by_state(**kw):
    """
    Build an HSAR setup with state-level spatially-autocorrelated random effects
    and county level observations
    """
    W = ps.queen_from_shapefile(ps.examples.get_path('south.shp'))
    try:
       from pysal.contrib import shapely_ext as sh
    except ImportError:
        Warn('Cannot import shapely, cannot build experiment')
    df = ps.pdio.read_files(ps.examples.get_path('south.shp'))
    gb = df.groupby('STATE_NAME')
    state_geoms = gb.geometry.apply(sh.unary_union)
    state_fips = gb.STATE_FIPS.mean()
    state_name = gb.STATE_NAME.max()

    outdf = pd.DataFrame([state_geoms, state_fips, state_name]).T
    ps.pdio.write_files('./tmp.shp')
    M = ps.queen_from_shapefile('./tmp.shp')

    M.transform = W.transform = 'r'

    state_names = df.STATE_NAME.unique()
    membership = df.STATE_NAME.apply(lambda x: list(state_names).index(x))

    return scenario_from_membership(membership, W, M, **kw)

def scenario_from_membership(membership, W, M, **kw):
    membership = np.asarray(membership).flatten()
    J = len(np.unique(membership))
    N = len(membership)
    Delta, mems = Delta_members(Delta=None, membership=membership, N=N, J=J)
    np.testing.assert_array_equal(mems, membership)
    Betas, Sigma2_e, Sigma2_u, Rho, Lambda = setup_params(**kw)
    y, X = dgp(Betas, Sigma2_e, Sigma2_u, Rho, W, Lambda, M, Delta) 
    return y, X, Betas, Sigma2_e, Sigma2_u, Rho, W, Lambda, M, Delta

def setup_params(**kw):
    Betas = kw.pop('Betas', None)
    if Betas is None:
        Betas = np.array([[(-1)**(i%2) * 2*i for i in range(1,5)]]).T
    Sigma2_e, Sigma2_u = kw.pop('Sigma2_e',None), kw.pop('Sigma2_u', None)
    if Sigma2_e is None:
        Sigma2_e = 2
    if Sigma2_u is None:
        Sigma2_u = 5
    Rho, Lambda = kw.pop('Rho',None), kw.pop('Lambda', None)
    if Rho is None:
        Rho = .5
    if Lambda is None:
        Lambda = -.2
    return Betas, Sigma2_e, Sigma2_u, Rho, Lambda
