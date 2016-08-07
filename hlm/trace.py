import pandas as pd
from .plotting.traces import plot_trace as _plt
import numpy as np
from six import iteritems as diter

PUBLIC_DICT_ATTS = [k for k in dir(dict) if not k.startswith('_')]

class Trace(dict):
    """
    This is a proxy class to add stuff to help with composition. It will expose
    dictionary methods directly to the class's __dict__, meaning it will work
    like a dot-access dictionary. 
    """
    def __init__(self, **kwargs):
        collisions = [k in PUBLIC_DICT_ATTS for k in kwargs.keys()]
        collisions = [k for k,collide in zip(kwargs.keys(), collisions) if collide]
        if len(collisions) > 0:
            raise TypeError('Passing {} to Trace will overwrite builtin dict methods. Bailing...'.format(collisions))
        self.__dict__.update(kwargs)
        self._dictify()

    def _dictify(self):
        """
        hack to make Trace pass as if it were a dict by passing methods
        straight through to its own dict
        """
        for method in PUBLIC_DICT_ATTS:
            if method is 'clear':
                continue #don't want to break the namespace
            self.__dict__[method] = eval('self.__dict__.{}'.format(method))

    @property
    def _data(self):
        return {k:v for k,v in diter(self.__dict__) if k not in PUBLIC_DICT_ATTS}

    def __repr__(self):
        innards = ', '.join(['{}:{}'.format(k,v) for k,v in diter(self._data)])
        return '{%s}' % innards
    
    def __getitem__(self, val):
        """
        passthrough to self.__dict__[val]
        """
        return self.__dict__[val]
    
    def __setitem__(self, key, item):
        self.__dict__[key] = item
    
    def __delitem__(self, key):
        del self.__dict__[key]
        self._dictify()

    def clear(self):
        not_builtins = {k for k in self.keys() if k not in PUBLIC_DICT_ATTS}
        for key in not_builtins:
            del self.__dict__[key]
    
    @property
    def varnames(self):
        return list(self._data.keys())

    def to_df(self):
        df = pd.DataFrame().from_records(trace._data)
        for col in df.columns:
            if isinstance(df[col][0], np.ndarray):
                # a flat nested (n,) of (u,) elements hstacks to (u,n)
                new = np.hstack(df[col].values)

                if new.shape[0] is 1:
                    newcols = [col]
                else:
                    newcols = [col + '_' + str(i) for i in range(new.shape[0])]
                # a df is (n,u), so transpose and DataFrame
                new = pd.DataFrame(new.T, columns=newcols)
                df.drop(col, axis=1, inplace=True)
                df = pd.concat((df[:], new[:]), axis=1)
        return df

    plot = _plt
