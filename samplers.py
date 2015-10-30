import numpy as np
from numpy import linalg as la
from scipy import stats
from warnings import warn
import copy
import clogposts as clp
from trace import Trace
from six import iteritems as diter

class Gibbs(object):
    """
    A Gibbs Sampler manager
    """
    def __init__(self, n=100, backend='', statics=globals(), **kwargs):
        self.var_names = [k for k in kwargs.keys()]
        self.trace = Trace(self.var_names, n, statics=statics)
        self.samplers = [v(self.trace) for v in kwargs.values()]
        self.step = 0
        self.allocated = n
        self.pos = self.trace.pos
        if backend is not '':
            self.backend = open(backend, 'w')
            self.backend.write(','.join([str(name) for name in self.var_names]))
            self.backend.write('\n')
        else:
            self.backend = None

    def reorder_step(self, order):
        inn = [x in order for x in self.var_names]
        out = [x not in self.var_names for x in order]
        if not all(inn): #if all order in var_names
            v = order[inn.index(False)]
            raise Exception('Variable {} in ordering is not registered.'.format(v))
        if any(out): #if any var_name not in order
            v = self.var_names[out.index(False)]
            raise Exception('Variable {} in registered variables has been omitted'.format(v))
        new_samplers = [None] * len(self.var_names)
        for new_i, name in enumerate(order):
            old_i = self.var_names.index(name)
            new_samplers[new_i] = self.samplers[old_i]
        self.var_names = order
        self.samplers = new_order

    def add_step(self, sampler, name, i=-1):
        self.var_names.insert(name, i)
        self.samplers.insert(sampler, i)
    
    def add_steps(self, *args):
        nsteps = len(args) // 2
        for i in range(nsteps):
            sampler = args[2*i]
            name = args[2*i+1]

    def drop_step(self, name=None, i=None):
        if name is not None:
            i = self.var_names.index(name)
        _ = self.var_names.pop(i)
        _ = self.samplers.pop(i)

    def drop_steps(self, *args):
        for arg in args:
            if isinstance(str, arg):
                self.drop_step(name=arg)
            else:
                self.drop_step(i=arg)

    def sample(self, steps = 1, n = None, verbose=True):
        if n is None:
            n = steps * len(self.var_names)
        if n % len(self.var_names) != 0:
            warn("Sampling {n} steps will stop inside of a full iteration!".format(n=n))
        if verbose:
            print("Sampling {pos}:{step}".format(pos=self.pos, step=self.step))
        for _ in range(n):
            self.next()
        if self.step is 0 and self.backend is not None:
            pt = self.trace.front()
            self.backend.write(','.join([str(pt[k]) for k in self.var_names]))
            self.backend.write('\n')
        if self.backend is not None:
            #only keep current state
            self.trace.Stochastics = self.trace.front()
    
    def next(self):
        if self.pos > self.allocated:
            raise StopIteration()
        self.samplers[self.step].next()
        self.step += 1
        self.step %= len(self.samplers)

    def __next__(self):
        return self.next()

    def __del__(self):
        self.backend.close()

class AbstractSampler(object):
    """
    One sampling step. This class is not intended to be used, but should be
    inherited. 

    The main idea is that classes that inherit from this will define some
    _cpost function that draws from a conditional posterior. 
    """
    def __init__(self, stochs, statics, exports, trace):
        self.stochs = stochs
        self.statics = statics
        self.exports = exports

    def next(self):
        """
        equivalent to calling the sample method
        """
        return self.__next__()

    def __next__(self):
        """
        equivalent to calling the sample method
        """
        return self.sample()

    def sample(self, trace=None, inplace=True):
        """
        return a new sample from the conditional posterior function, _cpost,
        using the most recent Random Variables in the attached trace. 
        """
        if trace is None:
            trace = self.trace
        if not inplace:
            return self._cpost(copy.copy(trace), inplace=False)
        else:
            return self._cpost(trace) #this is where the sampling function goes. 
    
    def _cpost(self):
        """
        The conditional posterior to sample from. Makes no sense for the
        abstract class
        """
        return np.random.random()

class Betas(AbstractSampler):
    """
    Sampler for the full beta conditional posterior in a HSAR model. 

    These are the "lower level" covariate parameters in a Hierarchical Simultaneous
    Autoregressive Model
    """
    def __init__(self, trace):
        self.stochs = ['sigma_e', 'rho', 'thetas']
        self.required = ['XtX', 'In', 'W', 'invT0', 'y', 'X', 'T0M0', 'Delta', 'p']
        self.exports = ['Ay', 'A']
        self.trace = trace

    def _cpost(self):
        """
        Full conditional posterior for Beta, as defined in Equation 26 of Dong
        & Harris (2014).
        """
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in self.trace.Statics:
                exec("{n} = s['{n}']".format(n=name))
            elif name in self.trace.Derived:
                exec("{n} = d['{n}']".format(n=name))
            else:
                raise KeyError("Variable {n} not found in trace".format(n=name))
        pt = self.trace.front() #grab most current sampled values
        VV = XtX / pt['sigma_e'] + invT0
        v_betas = la.inv(VV) #conditional posterior variance matrix
        A = In - pt['rho'] * W
        Ay = np.dot(A, y)
        Delta_u = np.dot(Delta, pt['thetas']) #recall, HSAR.R labels Delta from paper as Z
        lprod = np.dot(X.T, (Ay - Delta_u)/pt['sigma_e'] + T0M0) 
        m_betas = np.dot(v_betas, lprod) #conditional posterior mean
        new_betas = np.random.multivariate_normal(m_betas.flatten(), v_betas)
        new_betas = new_betas.reshape(pt['betas'].shape)
        self.trace.update('betas', new_betas) #update in place
        for name in self.exports:
            d[name] = eval(name)

class Thetas(AbstractSampler):
    """
    Sampler for the full Theta conditional poster in an HSAR model

    These are the "upper-level" random effects for a Hierarchical Simulatenous
    Autoregressive Model
    """
    def __init__(self, trace):
        self.stochs = ['lam', 'sigma_e', 'sigma_u', 'betas']
        self.required = ['Ij', 'M', 'X', 'y', 'J', 'Ay', 'Delta']
        self.exports = ['Xb', 'B']
        self.trace = trace
    
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
                raise KeyError("Variable {n} not found in trace".format(n=name))
        pt = self.trace.front()
        B = Ij - pt['lam'] * M
        v_u = np.dot(Delta.T, Delta)/pt['sigma_e'] + np.dot(B.T, B)/pt['sigma_u']
        v_u = la.inv(v_u) #conditional posterior variance matrix
        Xb = np.dot(X, pt['betas'])
        lprod = np.dot(Delta.T, Ay - Xb) / pt['sigma_e']
        m_u = np.dot(v_u, lprod)
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
    def __init__(self, trace):
        self.stochs = ['thetas']
        self.required = ['Delta', 'Ay', 'Xb', 'ce', 'd0']
        self.exports = ['Delta_u']
        self.trace = trace

    def _cpost(self):
        """
        Full conditional posterior for sigma_e as defined in equation 30 in Dong
        & Harris (2014)
        """
        pt = self.trace.front()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                raise KeyError("Variable {n} not found in trace".format(n=name))
        Delta_u = np.dot(Delta, pt['thetas'])
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
    def __init__(self, trace):
        self.stochs = ['thetas']
        self.required = ['B', 'b0', 'au']
        self.exports = []
        self.trace = trace

    def _cpost(self):
        """
        Full conditional posterior for sigma_u as defined in equation 29 of
        Dong & Harris (2014)
        """
        pt = self.trace.front()
        s = self.trace.Statics
        d = self.trace.Derived
        for name in self.required:
            if name in s:
                exec("{n} = s['{n}']".format(n=name))
            elif name in d:
                exec("{n} = d['{n}']".format(n=name))
            else:
                raise KeyError("Variable {n} not found in trace".format(n=name))
        Bus = np.dot(B, pt['thetas'])
        bu = .5 * np.dot(Bus.T, Bus) + b0
        new_sigma_u = stats.invgamma.rvs(au, scale=bu)
        self.trace.update('sigma_u', new_sigma_u)
        for name in self.exports:
            self.trace.Derived[name] = eval(name)

class Lambda(AbstractSampler):
    """
    Sampler for the full lambda conditional posterior of an HSAR model

    This is the "upper-level" error spatial dependence coefficient in a Hierarchical
    Simultaneous Autoregressive model
    """
    def __init__(self, trace):
        self.trace = trace

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for lambda as defined
        in equation 32 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        self.trace.update('lam', np.random.random()*2-1)

class Rho(AbstractSampler):
    """
    Sampler for the full rho conditional posterior of an HSAR model

    This is the "lower-level" response spatial dependence coefficient in a
    Hierarchical Simultaneous Autoregressive model
    """
    def __init__(self, trace):
        self.trace = trace

    def _cpost(self):
        """
        Will be the full conditional posterior distribution for rho as defined
        in equation 31 of Dong & Harris (2014), but is currently Unif(-1,1). 
        """
        self.trace.update('rho', np.random.random()*2-1)
