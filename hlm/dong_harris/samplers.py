import numpy as np
from numpy import linalg as la
import scipy.linalg as scla
from scipy import stats
from warnings import warn as Warn
from six import iteritems as diter

from pysal.spreg.diagnostics import constant_check
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot

from ..abstracts import Abstract_Step, Metropolis_Mixin
from ..utils import inversion, splogdet

__all__ = ['Betas', 'Thetas', 'Sigma2_e', 'Sigma2_u', 'Lambda', 'Rho']

class Betas(Abstract_Step):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, sampler=None, initial=None, name=None, method='mvn'):
        super(Betas, self).__init__(sampler=sampler, name=name) 
        if initial is None:
            try:
                p = self.sampler['p']
                initial = np.zeros((1,p))
            except TypeError:
                Warn('Initial default guess was not found in state')
        self.initial = initial
        self.method = method

    def _new_params(self):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014), eventually  a multivariate normal:
        
        vcov: ((sigma_e**-2) * X.T @ X + invT0)**(-1)
        Mean: vcov @ (sigma_e**-2) * (X.T @ (A @ y - Delta @ theta) + invT0 @ M0))

        """
        st = self.sampler._state #alias for shared state of sampler
        VV = st.XtX / st.Sigma2_e + st.invT0
        st.v_betas = scla.inv(VV) #conditional posterior variance matrix
        st.A = np.asarray(st.In - st.Rho * st.W)
        st.Ay = spdot(st.A, st.y)
        st.Delta_u = spdot(st.Delta, st.Thetas) #recall, HSAR.R mislabels Delta->Z
        lprod = spdot((st.Ay - st.Delta_u).T/st.Sigma2_e, st.X)
        lprod += st.T0M0.T
        st.m_betas = spdot(st.v_betas, lprod.T) #conditional posterior mean

    def _chol(self):
        st = self.sampler._state
        self._new_params()
        new_betas = spdot(np.random.normal(0, 1, size=st.p).reshape(1,st.p), 
                           scla.cholesky(st.v_betas))
        new_betas += st.m_betas.T
        st.Betas = new_betas.reshape(st.Betas.shape)
        return st.Betas
    
    def _mvn(self):
        st = self.sampler._state
        self._new_params()
        new_betas = np.random.multivariate_normal(st.m_betas.flatten(),
                st.v_betas)
        st.Betas = new_betas.reshape(st.Betas.shape)
        return st.Betas
    
    def _draw(self):
        if self.method.lower() in ['chol', 'cholesky']:
            return self._chol()
        else:
            return self._mvn()

class Thetas(Abstract_Step):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, sampler=None, initial=None, name=None, method='mvn'):
        super(Thetas, self).__init__(sampler=sampler, name=name) 
        if initial is None:
            try:
                J = self.sampler['J']
                initial = np.zeros((J,1))
            except KeyError:
                Warn('Initial default guess was not found in state')
        self.initial = initial
        self.method = method

    def _new_params(self):
        """
        Full conditional posterior for Theta, as defined in equation 28 in Dong
        & Harris (2014), eventually multivariate normal:

        vcov: [(sigma_e**-2) Delta.T @ Delta + (sigma_u**-2)B.T @ B]**(-1_
        mean: vcov @ [sigma_e**-2 Delta.T @ (A @ y - X @ Beta)]
        """
        st = self.sampler._state
        st.B = np.asarray(st.Ij - st.Lambda * st.M) #upper-level laplacian
        tmpv_u = spdot(st.Delta.T, st.Delta)/st.Sigma2_e 
        tmpv_u += spdot(st.B.T, st.B)/st.Sigma2_u
        st.v_u = scla.inv(tmpv_u) #conditional posterior variance matrix
        st.Xb = spdot(st.X, st.Betas.T) #uses recent copy of betas
        lprod = spdot(st.Delta.T, st.Ay - st.Xb) / st.Sigma2_e
        st.m_u = spdot(st.v_u, lprod) #conditional posterior means
        
    def _mvn(self):
        """
        Direct draw from the multivariate distribution after updating parameters
        """
        st = self.sampler._state
        self._new_params()
        new_u = np.random.multivariate_normal(st.m_u.flatten(), st.v_u)
        st.Thetas = new_u.reshape(st.Thetas.shape)
        return st.Thetas

    def _chol(self):
        """
        Use Cholesky covariance filtering to draw from the multivariate normal
        distribution after updating parameters. 
        """
        st = self.sampler._state
        self._new_params()
        new_u = spdot(np.random.normal(0,1,size=st.J).reshape(st.J,1).T,
                      scla.cholesky(st.v_u))
        new_u += st.m_u.T
        st.Thetas = new_u.reshape(st.Thetas.shape)
        return st.Thetas
    
    def _draw(self):
        """
        How to draw the next parameter value
        """
        if self.method.lower().startswith('cho'):
            return self._chol()
        else:
            return self._mvn()

class Sigma2_e(Abstract_Step):
    """
    Sampler for the full Sigma2_e conditional posterior in an HSAR model

    This is the variance for the "lower-level" errors in a Hierarhical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, sampler=None, initial=2, name=None):
        super(Sigma2_e, self).__init__(sampler=sampler, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial
    
    def _draw(self):
        """
        Full conditional posterior for Sigma2_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        st = self.sampler._state
        e = st.Ay - st.Delta_u - st.Xb
        st.de = .5 * spdot(e.T, e) + st.d0
        st.Sigma2_e = stats.invgamma.rvs(st.ce, scale=st.de)
        return st.Sigma2_e

class Sigma2_u(Abstract_Step):
    """
    Sampler for the full Sigma2_u conditional posterior in an HSAR model

    This is the variance for the "upper-level" random effects in a Hierarchical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, sampler=None, initial=2, name=None):
        super(Sigma2_u, self).__init__(sampler=sampler, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial

    def _draw(self):
        """
        Full conditional posterior for Sigma2_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        st = self.sampler._state
        Bus = spdot(st.B, st.Thetas)
        st.bu = .5 * spdot(Bus.T, Bus) + st.b0
        st.Sigma2_u = stats.invgamma.rvs(st.au, scale=st.bu)
        return st.Sigma2_u

class Lambda(Abstract_Step, Metropolis_Mixin):
    """
    Sampler for the full conditional of the spatially-aucorrelated 
    error parameter, called lambda.  

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, sampler=None, initial=.5, name=None, method='grid',
                 default_move=.1, adapt_rate=1., lower_bound=.4, upper_bound=.6,
                 proposal=stats.norm):
        Abstract_Step.__init__(self, sampler=sampler, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial
        Metropolis_Mixin.__init__(self, default_move = default_move,
                                  adapt_rate=adapt_rate,
                                  lower_bound=lower_bound, upper_bound=.6)
        self.method = method
    
    def _draw(self):
        if self.method.lower().startswith('met') or self.method.lower() == 'mh':
            return self._mh()
        else:
            return self._grid_inversion()

    def _mh(self):
        """
        A metropolis-hastings sampling strategy similar to the one employed in
        Lacombe & McIntyre (2015)
        """
        self.sampler._state.Lambda = self._metropolis(self.sampler._state.Lambda) 
        return self.sampler._state.Lambda

    def _grid_inversion(self):
        """
        Will be the full conditional posterior distribution for sac_upper as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        st = self.sampler._state
        parvals, logdets = st.Lambda_grid
        kernels = np.asarray([self._logp_kernel(val) 
                              for val in parvals]).flatten()

        log_density = logdets - kernels
        log_density = log_density - log_density.max()

        st.density = np.exp(log_density)
        
        st.Lambda = inversion(st.density, grid=parvals)
        return st.Lambda

    def _logp(self, val):
        if not (-1 < val < 1):
            return 0
        st = self.sampler._state
        kernel = self._logp_kernel(val)
        return splogdet(st.Ij - val * st.M) + kernel

    def _logp_kernel(self, val):
        if not (-1 < val < 1):
            return 0
        st = self.sampler._state

        uu = spdot(st.Thetas.T, st.Thetas)
        uMu = spdot(spdot(st.Thetas.T, st.M), st.Thetas)
        Mu = spdot(st.M, st.Thetas)
        uMMu = spdot(Mu.T, Mu)

        StS = uu - 2*val*uMu + uMMu * val**2
        return StS / (2*st.Sigma2_u)
    
    def _trunc_propose(self, current):
        """
        compute proposal & log transition probabilities for sampler at a given
        value.
        """
        a = (-1 - current) / self._move_size
        b = (1  - current) / self._move_size
        new = stats.truncnorm.rvs(a, b, loc=current, scale=self._move_size)
        new_a = (-1 - new)/ self._move_size
        new_b = (1 - new)/ self._move_size
        forward = stats.truncnorm.logpdf(new, a, b, 
                                         loc=current, scale=self._move_size)
        backward = stats.truncnorm.logpdf(current, new_a,new_b, 
                                         loc=new, scale=self._move_size)
        return new, forward, backward

class Rho(Abstract_Step, Metropolis_Mixin):
    """
    Sampler for the full Rho conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, sampler=None, initial=.5, name=None, method='grid',
                default_move=.1, adapt_rate=1., lower_bound=.4, upper_bound=.6,
                 proposal=stats.norm):
        Abstract_Step.__init__(self, sampler=sampler, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial
        Metropolis_Mixin.__init__(self, default_move = default_move,
                                  adapt_rate=adapt_rate,
                                  lower_bound=lower_bound, upper_bound=.6)
        self.method = method
    
    def _draw(self):
        if self.method.lower().startswith('met') or self.method.lower() == 'mh':
            return self._mh()
        else:
            return self._grid_inversion()

    def _mh(self):
        """
        A metropolis-hastings sampling strategy similar to the one employed in
        Lacombe & McIntyre (2015)
        """
        self.sampler._state.Rho = self._metropolis(self.sampler._state.Rho) 
        return self.sampler._state.Rho

    def _grid_inversion(self):
        """
        The gridded gibbs sampling strategy employed in Smith & LeSage (2004),
        adapted from Dong \& Harris (2014). 
        """
        st = self.sampler._state
        parvals, logdets = st.Rho_grid #values of sac_lower
       
        kernels = np.asarray([self._logp_kernel(val) 
                              for val in parvals]).flatten()
        log_density = logdets - kernels
        log_density = log_density - log_density.max()

        st.density = np.exp(log_density)

        st.Rho = inversion(st.density, grid=parvals)
        return st.Rho

    def _logp_kernel(self, val):
        """
        The kernel of the log conditional posterior of Rho. This is just the
        variance-scaled sum of squares term, no determinant. 
        """
        if not (-1 < val < 1):
            return 0
        st = self.sampler._state
        beta_u = spdot(st.XtXi, spdot(st.X.T, st.Delta_u))
        eu = st.Delta_u - spdot(st.X, beta_u)
        S = st.e0 - val * st.ed - eu
        StS = spdot(S.T, S) / (2 * st.Sigma2_e)
        return StS

    def _logp(self, val):
        """
        The full log posterior of rho. This is the sum of squares term plus the
        log determinant.
        """
        if not (-1 < val < 1):
            return 0
        st = self.sampler._state
        kernel = self._logp_kernel(val)
        return splogdet(st.In - val * st.W) + kernel

    def _trunc_propose(self, current):
        """
        compute proposal & log transition probabilities for sampler at a given
        value.
        """
        a = (-1 - current) / self._move_size
        b = (1  - current) / self._move_size
        new = stats.truncnorm.rvs(a, b, loc=current, scale=self._move_size)
        new_a = (-1 - new)/ self._move_size
        new_b = (1 - new)/ self._move_size
        forward = stats.truncnorm.logpdf(new, a, b, 
                                         loc=current, scale=self._move_size)
        backward = stats.truncnorm.logpdf(current, new_a,new_b, 
                                         loc=new, scale=self._move_size)
        return new, forward, backward

if __name__ == '__main__':
    samplers = []
    samplers.append(Betas())
    
