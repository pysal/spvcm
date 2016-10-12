from ..generic import Base_Generic
from ...utils import se_covariance
from ... import verify
import numpy as np

class Base_SESE(Base_Generic):
    def __init__(self, Y, X, W, M, Delta,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 truncation=None):
        super(Base_SESE, self).__init__(Y, X, W, M, Delta,
                                        n_samples=0, n_jobs=n_jobs,
                                        extra_traced_params=extra_traced_params,
                                        priors=priors,
                                        configs=configs,
                                        starting_values=starting_values,
                                        truncation=truncation)
        self.state.Psi_1 = se_covariance
        self.state.Psi_2 = se_covariance
        
        try:
            self.sample(n_samples, n_jobs =n_jobs)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))

class SESE(Base_SESE):
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', verbose=False,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 truncation=None,
                 center=True,
                 scale=False):
        W,M = verify.weights(W, M, transform=transform)
        self.M = M
        N,_ = X.shape
        J = M.n
        Mmat = M.sparse
        Wmat = W.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        if center:
            X = verify.center(X)
        if scale:
            X = verify.scale(X)


        X = verify.covariates(X)
        
        self._verbose = verbose

        super(SESE, self).__init__(Y, X, Wmat, Mmat, Delta,
                                   n_samples=n_samples,
                                   n_jobs = n_jobs,
                                   extra_traced_params=extra_traced_params,
                                   priors=priors,
                                   configs=configs,
                                   starting_values=starting_values,
                                   truncation=truncation)
