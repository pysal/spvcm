from __future__ import division

import copy
import numpy as np
from numpy import linalg as la
from scipy import stats
from warnings import warn
from six import iteritems as diter

from trace import Trace
from samplers import AbstractSampler
from utils import logdet, inversion_sample as isamp

__all__ = ['Beta', 'Theta', 'Sigma_e', 'Sigma_u', 'SAC_Lower', 'SAC_Upper']

class Beta(AbstractSampler):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the "lower level" covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, trace, initial=None):
        self.stochs = ['sigma_e', 'sac_lower', 'thetas']
        self.required = ['XtX', 'In', 'W', 'invT0', 'y', 'X', 'T0M0', 'Delta', 'p']
        self.exports = ['Ay', 'A', 'v_betas', 'm_betas']
        self.trace = trace
        if initial is None:
            _,p = self.trace.Statics['X'].shape 
            initial = np.zeros((1,p)) #from Dong & Harris (2014)
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014).
        """
        for name in self.required:
            if name in self.trace.Statics:
                exec("{n} = self.trace.Statics['{n}']".format(n=name))
            elif name in self.trace.Derived:
                exec("{n} = self.trace.Derived['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        pt = self.trace.front() #grab most current sampled values
        VV = XtX / pt['sigma_e'] + invT0
        v_betas = la.inv(VV) #conditional posterior variance matrix
        A = In - pt['sac_lower'] * W
        Ay = np.dot(A, y)
        Delta_u = np.dot(Delta, pt['thetas']) #recall, HSAR.R labels Delta from paper as Z
        lprod = np.dot(X.T, (Ay - Delta_u)/pt['sigma_e']) + T0M0.reshape(p,1)
        m_betas = np.dot(v_betas, lprod) #conditional posterior mean
        new_betas = np.random.multivariate_normal(m_betas.flatten(), v_betas)
        new_betas = new_betas.reshape(pt['betas'].shape)
        self.trace.update('betas', new_betas) #update in place
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class Theta(AbstractSampler):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, trace, initial=None):
        self.stochs = ['lam', 'sigma_e', 'sigma_u', 'betas']
        self.required = ['Ij', 'M', 'X', 'y', 'J', 'Ay', 'Delta']
        self.exports = ['Xb', 'B', 'm_u', 'v_u']
        self.trace = trace
        if initial is None:
            J, _ = self.trace['Delta'].shape
            initial = np.zeros((J,1))
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for Theta, as defined in equation 28 in Dong
        & Harris (2014)
        """
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        pt = self.trace.previous()
        betas_now = self.trace.current()['betas']
        B = Ij - pt['lam'] * M #upper-level laplacian
        v_u = np.dot(Delta.T, Delta)/pt['sigma_e'] + np.dot(B.T, B)/pt['sigma_u']
        v_u = la.inv(v_u) #conditional posterior variance matrix
        Xb = np.dot(X, betas_now.T) #uses recent copy of betas
        lprod = np.dot(Delta.T, Ay - Xb) / pt['sigma_e']
        m_u = np.dot(v_u, lprod) #conditional posterior means
        new_u = np.random.multivariate_normal(m_u.flatten(), v_u)
        new_u = new_u.reshape(pt['thetas'].shape)
        self.trace.update('thetas', new_u)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class Sigma_e(AbstractSampler):
    """
    Sampler for the full sigma_e conditional posterior in an HSAR model

    This is the variance for the "lower-level" errors in a Hierarhical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, trace, initial=2):
        self.stochs = ['thetas']
        self.required = ['Delta', 'Ay', 'Xb', 'ce', 'd0']
        self.exports = ['Delta_u', 'de']
        self.trace = trace
        self.initial = initial
    def _cpost(self):
        """
        Full conditional posterior for sigma_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        pt = self.trace.previous()
        cpt = self.trace.front()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        Delta_u = np.dot(Delta, cpt['thetas'])
        e = Ay - Delta_u - Xb
        de = .5 * np.dot(e.T, e) + d0
        new_sigma_e = stats.invgamma.rvs(ce, scale=de)
        self.trace.update('sigma_e', new_sigma_e)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class Sigma_u(AbstractSampler):
    """
    Sampler for the full sigma_u conditional posterior in an HSAR model

    This is the variance for the "upper-level" random effects in a Hierarchical
    Simulatenous Autoregressive model. 
    """
    def __init__(self, trace, initial=2):
        self.stochs = ['thetas']
        self.required = ['B', 'b0', 'au']
        self.exports = ['bu']
        self.trace = trace
        self.initial = initial

    def _cpost(self):
        """
        Full conditional posterior for sigma_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        pt = self.trace.previous()
        cpt = self.trace.current()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        Bus = np.dot(B, cpt['thetas'])
        bu = .5 * np.dot(Bus.T, Bus) + b0
        new_sigma_u = stats.invgamma.rvs(au, scale=bu)
        self.trace.update('sigma_u', new_sigma_u)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class SAC_Upper(AbstractSampler):
    """
    Sampler for the full sac_upper conditional posterior of an HSAR model

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, trace, initial=.5):
        self.trace = trace
        self.required = ["M", "sac_upper"]
        self.exports = [ 'parvals', 'density', 'S_sac_upper']
        self.initial = initial

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for sac_upper as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        pt = self.trace.previous()
        cpt = self.trace.current()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        parvals = sac_upper[:,0] #aka "lambda_exist"
        logdets = sac_upper[:,1] #aka "log_detlambda"
        nsac_upper = len(parvals)
        iota = np.ones_like(parvals)
        
        uu = np.dot(cpt['thetas'].T, cpt['thetas'])
        uMu = np.dot(np.dot(cpt['thetas'].T, M), cpt['thetas'])
        Mu = np.dot(M, cpt['thetas'])
        uMMu = np.dot(Mu.T, Mu)

        S_sac_upper = uu*iota - 2*parvals*uMu + uMMu*parvals**2

        log_density = logdets - S_sac_upper/(2*pt['sigma_u'])
        log_density = log_density - log_density.max()

        density = np.exp(log_density)
        
        new_sac_upper = isamp(density, grid=parvals) 
        self.trace.update('sac_upper', new_sac_upper)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class SAC_Lower(AbstractSampler):
    """
    Sampler for the full sac_lower conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, trace, initial=.5):
        self.trace = trace
        self.required = ["e0", "ed", "e0e0", "eded", "e0ed", "Delta_u", "X", "sac_lowers"]
        self.exports = ['parvals', 'density', 'S_sac_lower']
        self.initial = initial

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for sac_lower as defined
        in equation 31 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        pt = self.trace.previous()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                err = "Variable {} not found in trace".format(name)
                err += " at step {}".format(self.__class__)
                raise KeyError(err)
        parvals = sac_lowers[:,0] #values of sac_lower
        logdets = sac_lowers[:,1] #log determinant of laplacians
        nsac_lower = len(parvals)
        iota = np.ones_like(parvals)
        #have to compute eu-reltaed parameters
        beta_u, resids, rank, svs = la.lstsq(X, Delta_u)
        eu = Delta_u - np.dot(X, beta_u)
        eueu = np.dot(eu.T, eu)
        e0eu = np.dot(e0.T, eu)
        edeu = np.dot(ed.T, eu)

        S_sac_lower = (e0e0*iota + parvals**2 * eded + eueu 
                - 2*parvals*e0ed - 2*e0eu + 2*parvals*edeu)
        
        log_density = logdets - S_sac_lower/(2. * pt['sigma_e'])
        adj = log_density.max()
        log_density = log_density - adj #downshift to zero?

        density = np.exp(log_density)

        new_sac_lower = isamp(density, grid=parvals)
        self.trace.update('sac_lower', new_sac_lower)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)
