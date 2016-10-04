from ..generic import Generic, Base_Generic
from ...utils import se_covariance
from ... import verify
import numpy as np

class Base_SESE(Base_Generic):
    def __init__(self, Y, X, W, M, Delta, n_samples=1000, **configs):
        super(Base_SESE, self).__init__(Y, X, W, M, Delta, n_samples=0,
                                        skip_covariance=True, **configs)
        self.state.Psi_1 = se_covariance
        self.state.Psi_2 = se_covariance
        self._setup_covariance()
        try:
            self.sample(n_samples)
        except (np.linalg.LinAlgError, ValueError) as e:
            warn('Encountered the following LinAlgError. '
                 'Model will return for debugging purposes. \n {}'.format(e))

class SESE(Base_SESE):
    def __init__(self, Y, X, W, M, Z=None, Delta=None, membership=None,
                 #data options
                 transform ='r', n_samples=1000, verbose=False,
                 **options):
        W,M = verify.weights(W, M, transform=transform)
        self.M = M
        
        Y = Y - Y.mean() / Y.std()
        X = X - X.mean(axis=0) / X.std()
        
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
        super(SESE, self).__init__(Y, X, Wmat, Mmat, Delta, n_samples,
                                           **options)
