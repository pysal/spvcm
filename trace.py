from six import iteritems as diter
import copy

class Trace(object):
    def __init__(self, stochs, size, **kwargs):
        self.Stochastics = {k:[None]*size for k in stochs}
        self.Statics = kwargs.pop('statics', None)
        if self.Statics is None:
            self.Statics = globals() #just pull globals if statics isn't given
        self.Derived = {}
        self.pos = 0
        self.var_names = self.Stochastics.keys()
        self.const_names = self.Statics.keys()
    def update(self, name, val):
        if name not in self.var_names:
            raise KeyError("Variable {n} not found in variable list".format(n=name))
        if self.Stochastics[name][self.pos] is None:
            self.Stochastics[name][self.pos] = val
        else:
            stepcheck = [v[self.pos] is not None for v in self.Stochastics.values()]
            if all(stepcheck):
                self.pos += 1
                self.update(name, val)
                return val
            else:
                behind = stepcheck.index(False)
                print('Cowardly refusing to leave ' + self.var_names[behind] + ' behind')

    def point(self, idx):
        return {k:self.Stochastics[k][idx] for k in self.var_names}
    
    def current(self):
        return self.point(self.pos)

    def previous(self):
        return self.point(self.pos -1)

    def front(self):
        """
        get the current non-null values of the trace
        """
        cu = self.current()
        pre = self.previous()
        
        out = copy.copy(cu)

        for k,v in diter(out):
            if v is None:
                out[k] = pre[k]

        return out
