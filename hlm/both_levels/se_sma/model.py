from ..generic import Base_Generic
from ..generic.sample import logp_rho_prec
from ...utils import se_covariance, sma_covariance, se_precision, sma_precision
from ... import verify
import numpy as np

class Base_SESMA(Base_Generic):
    def __init__(self, Y, X, W, M, Delta,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None):
        super(Base_SESMA, self).__init__(Y, X, W, M, Delta,
                                        n_samples=0, n_jobs=n_jobs,
                                        extra_traced_params=extra_traced_params,
                                        priors=priors,
                                        configs=configs,
                                        starting_values=starting_values)
        st = self.state
        st.Psi_1 = se_covariance
        st.Psi_2 = sma_covariance
        self.state.Psi_1i = se_precision
        self.state.Psi_2i = sma_precision

        self.configs.Rho.logp = logp_rho_prec
        
        st.Lambda_min, st.Lambda_max = -st.Lambda_max, -st.Lambda_min
        try:
            self.sample(n_samples, n_jobs=n_jobs)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))


class SESMA(Base_SESMA):
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', verbose=False,
                 n_samples=1000, n_jobs=1,
                 extra_traced_params = None,
                 priors=None,
                 configs=None,
                 starting_values=None,
                 center=True,
                 scale=False):
        W,M = verify.weights(W,M, transform=transform)
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

        super(SESMA, self).__init__(Y, X, Wmat, Mmat, Delta,
                                   n_samples=n_samples,
                                   n_jobs = n_jobs,
                                   extra_traced_params=extra_traced_params,
                                   priors=priors,
                                   configs=configs,
                                   starting_values=starting_values)
