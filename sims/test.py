import hlm
from hlm import HSAR
from hlm.dong_harris.verify import Delta_members
from . import dgp
import pysal as ps
import pandas as pd
import numpy as np
import scipy.sparse as spar
from functools import partial
import itertools as it

PARAMETER_NAMES = ['Betas', 'Sigma2_e', 'Sigma2_u', 'Rho', 'Lambda']

W = ps.open('./test_data/w_lower.mtx').read()
M = ps.open('./test_data/w_upper.mtx').read()
df = pd.read_csv('./test_data/test.csv')
membership = df[['county']].values - 1
W.transform = M.transform = 'r'
Wmat = W.sparse
Mmat = M.sparse
Delta, mems = Delta_members(Delta = None, membership=membership, N=W.n, J=M.n)
W_min, W_max = hlm.utils.speigen_range(Wmat)
M_min, M_max = hlm.utils.speigen_range(Mmat)

rhogrid = np.arange(1./W_min, 1./W_max, .01)
lamgrid = np.arange(1./M_min, 1./M_max, .01)

In = np.identity(W.n)
Ij = np.identity(M.n)

As = [spar.csc_matrix(In - rho * Wmat) for rho in rhogrid]
Bs = [spar.csc_matrix(Ij - lam * Mmat) for lam in lamgrid]

rhodets = np.hstack([hlm.utils.splogdet(A) for A in As]).flatten()
lamdets = np.hstack([hlm.utils.splogdet(B) for B in Bs]).flatten()

rhos = np.hstack((rhogrid, rhodets))
lams = np.hstack((lamgrid, lamdets))

def exp_promise(**kw):
    Betas, Sigma2_e, Sigma2_u, Rho, Lambda = dgp.setup_params(**kw)
    y, X = dgp.dgp(Betas, Sigma2_e, Sigma2_u, Rho, W, Lambda, M, Delta)
    expgrd = partial(HSAR, y,X,W,M,membership=membership, n_samples=5000,
                     spatial_method='grid', effects_method='chol',
                     rho_grid = rhos, lambda_grid=lams,
                     truncate=(W_min, W_max, M_min, M_max))
    expmet = partial(HSAR, y,X,W,M,membership=membership, n_samples=5000,
                     spatial_method='met', effects_method='chol',
                     truncate=(W_min, W_max, M_min, M_max))
    ols = partial(ps.spreg.OLS, y,X)
    confs = [Betas, Sigma2_e, Sigma2_u, Rho, Lambda]
    return expgrd, expmet, ols, confs

def run_exp(pid, **configs):
    grid, met, ols, confs = exp_promise(**configs)
    grid = grid()
    met = met()
    ols = ols()
    outstr = '_'.join('{}-{:.3}'.format(k,v) 
                      for k,v in sorted(configs.keys()))

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

def build_frame(**kw):
    frame = dict()
    for param in PARAMETER_NAMES:
        conf = kw.get(param, None)
        if conf is not None:
            #conf should be (min,max,step) tuple or array of values
            if isinstance(conf, (np.ndarray, list)):
                print('interpreting {} as set of values'.format(param))
                frame.update({param:conf})
            else:
                print('interpreting {} as (min,max,step)'.format(param))
                frame.update({param:np.arange(*conf)})
    allexps = it.product(*frame.items())
    labelled = ({k:v for k,v in zip(allexps.keys(), exp)} 
                     for exp in allexps)
    return labelled
    for l in labelled:
        yield l

    

