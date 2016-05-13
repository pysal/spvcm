import numpy as np
from numpy import linalg as la
import scipy.linalg as scla
from scipy import stats
from warnings import warn as Warn
from six import iteritems as diter

from pysal.spreg.diagnostics import constant_check
from pysal.spreg.utils import spmultiply, sphstack, spmin, spmax, spdot, splogdet

from ..abstracts import AbstractSampler
from ..utils import inversion

__all__ = ['Betas', 'Thetas', 'Sigma2_e', 'Sigma2_u', 'Lambda', 'Rho']

class Betas(AbstractSampler):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None, name=None):
        super(Betas, self).__init__(state=state, name=name) 
        if initial is None:
            try:
                p = self.state['p']
                initial = np.zeros((1,p))
            except TypeError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014), eventually  a multivariate normal:
        
        vcov: ((sigma_e**-2) * X.T @ X + invT0)**(-1)
        Mean: vcov @ (sigma_e**-2) * (X.T @ (A @ y - Delta @ theta) + invT0 @ M0))

        """
        st = self.state._state #alias for shared state of sampler
        fr = self.state.front #alias for the most recent parameter values
        in equation 31 of Dong & Harris (2014). 
        VV = st.XtX / fr.Sigma2_e + st.invT0
        st.v_betas = scla.inv(VV) #conditional posterior variance matrix
        st.A = np.asarray(st.In - fr.Rho * st.W)
        st.Ay = spdot(st.A, st.y)
        st.Delta_u = spdot(st.Delta, fr.Thetas) #recall, HSAR.R mislabels Delta->Z
        lprod = spdot((st.Ay - st.Delta_u).T/fr.Sigma2_e, st.X)
        lprod += st.T0M0.T
        st.m_betas = spdot(st.v_betas, lprod.T) #conditional posterior mean
        new_betas = np.random.multivariate_normal(st.m_betas.flatten(), st.v_betas)
        st.Betas  = new_betas.reshape(fr.Betas.shape)
        return st.Betas

    def _cpost_chol(self):
        st = self.state._state #alias for shared state of sampler
        fr = self.state.front #alias for the most recent parameter values
        VV = st.XtX / fr.Sigma2_e + st.invT0
        st.v_betas = scla.inv(VV) #conditional posterior variance matrix
        st.A = np.asarray(st.In - fr.Rho * st.W)
        st.Ay = spdot(st.A, st.y)
        st.Delta_u = spdot(st.Delta, fr.Thetas) #recall, HSAR.R mislabels Delta->Z
        lprod = spdot(st.X.T, (st.Ay - st.Delta_u).T/fr.Sigma2_e)
        lprod += st.T0M0.T
        st.m_betas = spdot(st.v_betas, lprod.T) #conditional posterior mean
        new_betas = spdot(np.random.normal(0, 1, size=st.p).reshape(1,st.p), 
                           scla.cholesky(st.v_betas))
        new_betas += st.m_betas.T
        st.Betas = new_betas
        return st.Betas

class Thetas(AbstractSampler):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, state=None, initial=None, name=None):
        super(Thetas, self).__init__(state=state, name=name) 
        if initial is None:
            try:
                J = self.state['J']
                initial = np.zeros((J,1))
            except KeyError:
                Warn('Initial default guess was not found in state')
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Theta, as defined in equation 28 in Dong
        & Harris (2014), eventually multivariate normal:

        vcov: [(sigma_e**-2) Delta.T @ Delta + (sigma_u**-2)B.T @ B]**(-1_
        mean: vcov @ [sigma_e**-2 Delta.T @ (A @ y - X @ Beta)]
        """
        st = self.state._state
        fr = self.state.front
        st.B = np.asarray(st.Ij - fr.Lambda * st.M) #upper-level laplacian
        tmpv_u = spdot(st.Delta.T, st.Delta)/fr.Sigma2_e 
        tmpv_u += spdot(st.B.T, st.B)/fr.Sigma2_u
        st.v_u = scla.inv(tmpv_u) #conditional posterior variance matrix
        st.Xb = spdot(st.X, fr.Betas.T) #uses recent copy of betas
        lprod = spdot(st.Delta.T, st.Ay - st.Xb) / fr.Sigma2_e
        st.m_u = spdot(st.v_u, lprod) #conditional posterior means
        new_u = np.random.multivariate_normal(st.m_u.flatten(), st.v_u)
        st.Thetas = new_u.reshape(fr.Thetas.shape)
        return st.Thetas

    def _cpost_chol(self):
        st = self.state._state
        fr = self.state.front
        st.B = np.asarray(st.Ij - fr.Lambda * st.M) #upper-level laplacian
        tmpv_u = spdot(st.Delta.T, st.Delta)/fr.Sigma2_e 
        tmpv_u += spdot(st.B.T, st.B)/fr.Sigma2_u
        st.v_u = scla.inv(tmpv_u) #conditional posterior variance matrix
        st.Xb = spdot(st.X, fr.Betas.T) #uses recent copy of betas
        lprod = spdot(st.Delta.T, st.Ay - st.Xb) / fr.Sigma2_e
        st.m_u = spdot(st.v_u, lprod) #conditional posterior means
        new_u = spdot(np.random.normal(0,1,size=st.J).reshape(st.J,1).T,
                      scla.cholesky(st.v_u))
        new_u += st.m_u.T
        st.Thetas = new_u.reshape(fr.Thetas.shape)
        return st.Thetas

class Sigma2_e(AbstractSampler):
    """
    Sampler for the full Sigma2_e conditional posterior in an HSAR model

    This is the variance for the "lower-level" errors in a Hierarhical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2, name=None):
        super(Sigma2_e, self).__init__(state=state, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Sigma2_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        st = self.state._state
        e = st.Ay - st.Delta_u - st.Xb
        st.de = .5 * spdot(e.T, e) + st.d0
        st.Sigma2_e = stats.invgamma.rvs(st.ce, scale=st.de)
        return st.Sigma2_e

class Sigma2_u(AbstractSampler):
    """
    Sampler for the full Sigma2_u conditional posterior in an HSAR model

    This is the variance for the "upper-level" random effects in a Hierarchical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, state=None, initial=2, name=None):
        super(Sigma2_u, self).__init__(state=state, name=name) 
        if initial is None:
            initial = 2
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Sigma2_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        fr = self.state.front
        st = self.state._state
        Bus = spdot(st.B, fr.Thetas)
        st.bu = .5 * spdot(Bus.T, Bus) + st.b0
        st.Sigma2_u = stats.invgamma.rvs(st.au, scale=st.bu)
        return st.Sigma2_u

class Lambda(AbstractSampler):
    """
    Sampler for the full conditional of the spatially-aucorrelated 
    error parameter, called lambda.  

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5, name=None):
        super(Lambda, self).__init__(state=state, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for sac_upper as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        fr = self.state.front
        st = self.state._state
        parvals, logdets = st.Lambda_grid
        kernels = np.asarray([self._logp_kernel(val) 
                              for val in parvals]).flatten()

        log_density = logdets - kernels
        log_density = log_density - log_density.max()

        st.density = np.exp(log_density)
        
        st.Lambda = inversion(st.density, grid=parvals)
        return st.Lambda

    def _logp(self, val):
        st = self.state._state
        kernel = self._logp_kernel(val)
        return splogdet(st.In - val * st.M) + kernel

    def _logp_kernel(self, val):
        fr = self.state.front
        st = self.state._state

        uu = spdot(fr.Thetas.T, fr.Thetas)
        uMu = spdot(spdot(fr.Thetas.T, st.M), fr.Thetas)
        Mu = spdot(st.M, fr.Thetas)
        uMMu = spdot(Mu.T, Mu)

        StS = uu - 2*val*uMu + uMMu * val**2
        return StS / (2*fr.Sigma2_u)

class Rho(AbstractSampler, Metropolis_Mixin):
    """
    Sampler for the full Rho conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, state=None, initial=.5, name=None, method='grid',
                 default_move=1, adapt_rate=1, lower_bound=.4, upper_bound=.6,
                 proposal=stats.normal):
        AbstractSampler.__init__(state=state, name=name) 
        if initial is None:
            initial = .5
        self.initial = initial
        if method in ['mh', 'metropolis']:
            Metropolis_Mixin.__init__(default_move = default_move,
                                      adapt_rate=adapt_rate,
                                      lower_bound=lower_bound, upper_bound=.6)
            self._cpost = _mh

    def _mh(self):
        """
        A metropolis-hastings sampling strategy similar to the one employed in
        Lacombe & McIntyre (2015)
        """
        st.Rho = self._metropolis(st.Rho) 
        return st.Rho

    def _grid_inversion(self):
        """
        The gridded gibbs sampling strategy employed in Smith & LeSage (2004),
        adapted from Dong \& Harris (2014). 
        """
        fr = self.state.front
        st = self.state._state
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
        st = self.state._state
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
        st = self.state._state
        kernel = self._logp_kernel(val)
        return splogdet(st.In - val * st.W) + kernel

    def _propose(self, current, adapt_step=1):
        """
        compute proposal & log transition probabilities for sampler at a given
        value.
        """
        new = stats.truncnorm.rvs(-1, 1, loc=current, scale=1*adapt_step)
        forward = stats.truncnorm.logpdf(new, -1, 1, loc=current, scale=1*adapt_step)
        backward = stats.truncnorm.logpdf(current, -1,1, loc=new, scale=1*adapt_step)
        return new, forward, backward

         


if __name__ == '__main__':
    samplers = []
    samplers.append(Betas())
    
