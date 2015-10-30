from six import iteritems as diter
from warnings import warn
import copy

class Trace(object):
    """
    A trace object that contains the results of sampling runs.

    Deliberately duplicates some PyMC3 trace functionality to make it easy to
    transition this codebase to PyMC3. 

    Arguments:
    ============
    stochs  : names of Stochastic variables to track in the trace
    size    : length to allocate for the trace. 
    
    **kwargs : optional arguments
    ------------------------------
    statics : dictionary of names:values containing values that do not change
              during the trace. 

    Attributes:
    ===========
    Stochastics : dictionary of names & draws form a sampling procedure
    Statics     : dictionary of static precomputed values used in the sampling
    Derived     : dictionary of values that gets recomputed over the sample
    pos         : integer containing current position of the trace
    var_names   : names of random variables recorded in Stochastics
    const_names : names of static values recorded in Statics
    """
    def __init__(self, stochs, size, **kwargs):
        self.Stochastics = {k:[None]*size for k in stochs}
        self._prealloc = size
        self.Statics = kwargs.pop('statics', None)
        if self.Statics is None:
            self.Statics = globals() #just pull globals if statics isn't given
        self.Derived = {}
        self.pos = 0
        self.var_names = self.Stochastics.keys()
        self.const_names = self.Statics.keys()

    def update(self, name, val):
        """
        update the trace with a new value. If all RVs have been updated, this
        incremenets the trace position. 

        Arguments:
        ===========
        name    : name of variable in trace to update
        val     : value to add to the trace
        """
        if self.pos > self._alloc:
            warn("Sampling past preallocated space. Extending")
            self._extend(1)
            self._alloc += 1
        if name not in self.var_names:
            raise KeyError("Variable {n} not found in variable list".format(n=name))
        if self.Stochastics[name][self.pos] is None:
            try:
                self.Stochastics[name][self.pos] = val
            except IndexError:
                self.Stochatics[name].append(val)
        else:
            stepcheck = [v[self.pos] is not None for v in self.Stochastics.values()]
            if all(stepcheck):
                self.pos += 1
                self.update(name, val)
                return val
            else:
                behind = stepcheck.index(False)
                print('Cowardly refusing to leave ' + self.var_names[behind] + ' behind')
    
    def _extend(self, n): 
        """
        extend the trace allocation by n iterations
        """
        [self.Stochastics[name].extend([None] * n) for name in self.var_names]

    def point(self, idx):
        """
        returns a dictionary of names:values for Stochastics in the trace
        at the position `idx`. 
        """
        return {k:self.Stochastics[k][idx] for k in self.var_names}
    
    def current(self):
        """
        returns a dictionary of names:values for Stochastics in the trace
        in the current step. May be incomplete, and contain None values
        """
        return self.point(self.pos)

    def previous(self):
        """
        returns a dictionary of names:values for Stochastics in the trace
        in the previous step. Should always be complete, and contain no None
        values
        """
        return self.point(self.pos -1)

    def front(self):
        """
        get the most recent valid values of the trace.
        """
        cu = self.current()
        pre = self.previous()
        
        out = copy.copy(cu)

        for k,v in diter(out):
            if v is None:
                out[k] = pre[k]

        return out
