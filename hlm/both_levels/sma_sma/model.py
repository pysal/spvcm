from ..generic import Generic, Base_Generic
from ...utils import sma_covariance
from ... import verify
import numpy as np

class Base_SMASMA(Base_Generic):
    def __init__(self, Y, X, W, M, Delta, n_samples=1000, **configs):
        super(Base_SMASMA, self).__init__(Y, X, W, M, Delta, n_samples=0, 
                                        skip_covariance=True, **configs)
        st = self.state
        self.state.Psi_1 = sma_covariance
        self.state.Psi_2 = sma_covariance
        self._setup_covariance()
        st.Rho_min, st.Rho_max = -st.Rho_max, -st.Rho_min
        st.Lambda_min, st.Lambda_max = -st.Lambda_max, -st.Lambda_min
        try:
            self.sample(n_samples)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))

class SMASMA(Base_SMASMA):
    def __init__(self, y, X, M, W, Z=None, Delta=None, membership=None, 
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        M,W = verify.weights(M, W, transform=transform)
        self.M = M

        N,_ = X.shape
        J = M.n
        Mmat = M.sparse
        Wmat = W.sparse

        Delta, membership = verify.Delta_members(Delta, membership, N, J)

        X = verify.covariates(X)

        self._verbose = verbose
        if Z is not None:
            Z = Delta.dot(Z)
            X = np.hstack((X,Z))
        super(SMASMA, self).__init__(y, X, Wmat, Mmat, Delta, n_samples,
                                           **options)
