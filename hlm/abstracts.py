from warnings import warn as Warn
from datetime import datetime as dt
import numpy as np
import pysal as ps
import sqlite3 as sql
from .sqlite import head_to_sql, start_sql

class Sampler_Mixin(object):
    def __init__(self):
        pass

    def sample(self, n_samples):
        """
        Sample from the joint posterior distribution defined by all of the
        parameters in the gibbs sampler. 

        Parameters
        ----------
        n_samples      :   int
                        number of samples from the joint posterior density to
                        take

        Returns
        -------
        updates all values in place, may return trace of sampling run if pop is
        True
        """
        self._finalize_invariants()
        _start = dt.now()
        try:
            while n_samples > 0:
                if (self._verbose > 1) and (n_samples % 100 == 0):
                    print('{} Draws to go'.format(n_samples))
                self.draw()
                n_samples -= 1
        except KeyboardInterrupt:
            Warn('Sampling interrupted, drew {} samples'.format(self.cycles))
        finally:
            _stop = dt.now()
            if not hasattr(self, 'total_sample_time'):
                self.total_sample_time = _stop - _start
            else:
                self.total_sample_time += _stop - _start 

    def draw(self):
        """
        Take exactly one sample from the joint posterior distribution
        """
        self._sample()
        self.cycles += 1
        for param in self.traced_params:
            getattr(self.trace, param).append(getattr(self.state, param))
        if self.database is not None:
            head_to_sql(self, self._cur, self._cxn)
            for param in self.traced_params:
                self.trace[param] = [getattr(self.trace, param)[-1]]

    @property
    def database(self):
        return getattr(self, '_db', None)

    @database.setter
    def database(self, filename):
        self._cxn, self._cur = start_sql(self)
        self._db = filename

    @classmethod
    def from_other(cls, state, trace, model_type=None):
        if model_type == None:
            out = cls()
            out.state = state
            out.trace = trace
        else:
            X = np.zeros((9,1))
            Y = np.zeros((9,1))
            W = ps.lat2W(3,3)
            M = ps.lat2W(2,2)
            membership = np.array([0,0,1,1,2,2,2,3,3]),
            try:
                out = model_type(X, Y, 
                                 M=M, W=W, 
                                 membership=membership,
                                 n_samples=0)
            except TypeError:
                try:
                    out = model_type(X, Y, 
                                     M=M,  
                                     membership=membership,
                                     n_samples=0)
                except TypeError:
                    out = model_type(X, Y, 
                                     W=W,  
                                     membership=membership,
                                     n_samples=0)
            out.state = state
            out.trace = trace
        return out
    
    def deepcopy(self):
        return self.from_other(copy.deepcopy(self.state), 
                               copy.deepcopy(self.trace),
                               model_type = type(self))

def from_st(cls, state, trace):
    if model_type == None:
        out = cls()
        out.state = state
        out.trace = trace
    else:
        #build a dummy model
        X = np.zeros((9,1))
        Y = np.zeros((9,1))
        W = ps.lat2W(3,3)
        M = ps.lat2W(2,2)
        membership = np.array([0,0,1,1,2,2,2,3,3]),
        try:
            out = model_type(X, Y, 
                             M=M, W=W, 
                             membership=membership,
                             n_samples=0)
        except TypeError:
            try:
                out = model_type(X, Y, 
                                 M=M,  
                                 membership=membership,
                                 n_samples=0)
            except TypeError:
                out = model_type(X, Y, 
                                 W=W,  
                                 membership=membership,
                                 n_samples=0)
        out.state = state
        out.trace = Trace(**{k:[v] for k,v in trace._data.items()})
    return out
