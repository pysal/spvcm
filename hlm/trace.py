import pandas as pd
from .abstracts import Hashmap
from .plotting.traces import plot_trace as _plt
import numpy as np

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
        return {k:v for k,v in self.__dict__.items() if k not in PUBLIC_DICT_ATTS}

    def __repr__(self):
        innards = ', '.join(['{}:{}'.format(k,v) for k,v in self._data.items()])
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

    def point(i):
        """
        get the ith iteration in the trace.
        """
        out = Trace(**copy.copy({k:v[i] for k,v in self._data.items()}))
        return out

    def clear(self):
        not_builtins = {k for k in self.keys() if k not in PUBLIC_DICT_ATTS}
        for key in not_builtins:
            del self.__dict__[key]
    
    @property
    def varnames(self):
        return list(self._data.keys())

    def to_df(self):
        df = pd.DataFrame().from_records(self._data)
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
    
    def plot(self, *args, **kwargs):
        return _plt(None, trace=self, **kwargs)

class NewTrace(object):
    def __init__(self, *chains, **kwargs):
        if chains is () and kwargs != dict():
            self.chains = _maybe_hashmap(kwargs)
        elif chains is not ():
            self.chains = _maybe_hashmap(*chains)
            if kwargs != dict():
                self.chains.extend(_maybe_hashmap(kwargs))
        self._validate_schema()


    @property
    def varnames(self, chain=None):
        try:
            return self._varnames
        except AttributeError:
            try:
                self._validate_schema()
            except KeyError:
                if chain is None:
                    raise Exception('Variable names are heterogeneous in chains and no default index provided.')
                else:
                    warn('Variable names are heterogenous in chains!', stacklevel=2)
                    return list(self.chains[chain].keys())
            self._varnames = self.chains[0].keys()
            return self._varnames

    def _validate_schema(self):
        tracked_in_each = [set(chain.keys()) for chain in self.chains]
        same_schema = [names == tracked_in_each[0] for names in tracked_in_each]
        try:
            assert all(same_schema)
        except AssertionError:
            bad_chains = [i for i in range(self.n_chains) if same_schema[i]]
            KeyError('The parameters tracked in each chain are not the same!'
                     '\nChains {} do not have the same parameters as chain 1!'.format(bad_chains))
    
    
    @property
    def n_chains(self):
        return len(self.chains)
        
    def __getitem__(self, key):
        """
        Getting an item from a trace can be done using at most three indices, where:

        1 index
        --------
            str/list of str: names of variates in all chains to grab. Returns list of Hashmaps
            slice/int: iterations to grab from all chains. Returns list of Hashmaps, sliced to the specification

        2 index
        -------
            (str/list of str, slice/int): first term is name(s) of variates in all chains to grab, 
                                          second term specifies the slice each chain. 
                                          returns: list of hashmaps with keys of first term and entries sliced by the second term.
            (slice/int, str/list of str): first term specifies which chains to retrieve,
                                          second term is name(s) of variates in those chains
                                          returns: list of hashmaps containing all iterations
            (slice/int, slice/int): first term specifies which chains to retrieve,
                                    second term specifies the slice of each chain.
                                    returns: list of hashmaps with entries sliced by the second term
        3 index
        --------
            (slice/int, str/list of str, slice/int) : first term specifies which chains to retrieve,
                                                      second term is the name(s) of variates in those chains,
                                                      third term is the iteration slicing.
                                                      returns: list of hashmaps keyed on second term, with entries sliced by the third term
        """
        cls = type(self)
        if isinstance(key, str): #user wants a name from the traces
            if self.n_chains  > 1:
                return [chain[key] for chain in self.chains]
            else:
                return self.chains[0][key]
        elif isinstance(key, (slice, int)): #user wants all draws past a certain index
            if self.n_chains > 1:
                return [Hashmap(**{k:v[key] for k,v in chain.items()}) for chain in self.chains]
            else:
                return Hashmap(**{k:v[key] for k,v in self.chains[0].items()})
        elif isinstance(key, list) and all([isinstance(val, str) for val in key]): #list of atts over all iters and all chains
                if self.n_chains > 1:
                    return [Hashmap(**{k:chain[k] for k in key}) for chain in self.chains]
                else:
                    return Hashmap(**{k:self.chains[0][k] for k in key})
        elif isinstance(key, tuple): #complex slicing
            if len(key) == 1:
                return self[key[0]] #ignore empty blocks
            if len(key) == 2:
                head, tail = key
                if isinstance(head, str): #all chains, one var, some iters
                    if self.n_chains > 1:
                        return [_ifilter(tail, chain[head]) for chain in self.chains]
                    else:
                        return _ifilter(tail, self.chains[0][head])
                elif isinstance(head, list) and all([isinstance(v, str) for v in head]): #all chains, some vars, some iters
                    if self.n_chains > 1:
                        return [Hashmap(**{name:_ifilter(tail, chain[name]) for name in head}) 
                                   for chain in self.chains]
                    else:
                        chain = self.chains[0]
                        return Hashmap(**{name:_ifilter(tail, chain[name]) for name in head})
                elif isinstance(tail, str):
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    if len(target_chains) > 1:
                        return [chain[tail] for chain in target_chains]
                    elif len(target_chains) == 1:
                        return target_chains[0][tail]
                    else:
                        raise IndexError('The supplied chain index {} does not'
                                        ' match any chains in trace.chains'.format(head))
                elif isinstance(tail, list) and all([isinstance(v, str) for v in tail]):
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    if len(target_chains) > 1:
                        return [Hashmap(**{k:chain[k] for k in tail}) for chain in target_chains]
                    elif len(target_chains) == 1:
                        return Hashmap(**{k:target_chains[0][k] for k in tail})
                    else:
                        raise IndexError('The supplied chain index {} does not'
                                         ' match any chains in trace.chains'.format(head))
                else:
                    target_chains = _ifilter(head, self.chains)
                    if isinstance(target_chains, Hashmap):
                        target_chains = [target_chains]
                    out = [Hashmap(**{k:_ifilter(tail, val) for k,val in chain.items()})
                            for chain in target_chains]
                    if len(out) == 1:
                        return out[0]
                    else:
                        return out
            elif len(key) == 3:
                chidx, varnames, iters = key
                if isinstance(chidx, int):
                    if np.abs(chidx) > self.n_chains:
                        raise IndexError('The supplied chain index {} does not'
                                         ' match any chains in trace.chains'.format(chidx))
                if varnames == slice(None, None, None):
                    varnames = self.varnames
                chains = _ifilter(chidx, self.chains)
                if isinstance(chains, Hashmap):
                    chains = [chains]
                nchains = len(chains)
                if isinstance(varnames, str):
                    varnames = [varnames]
                if varnames is slice(None, None, None):
                    varnames = self.varnames
                if len(varnames) == 1:
                    if nchains > 1:
                        return [_ifilter(iters, chain[varnames[0]]) for chain in chains]
                    else:
                        return _ifilter(iters, chains[0][varnames[0]])
                else:
                    if nchains > 1:
                        return [Hashmap(**{varname:_ifilter(iters, chain[varname]) 
                                        for varname in varnames}) 
                                for chain in chains]
                    else:
                        return Hashmap(**{varname:_ifilter(iters, chains[0][varname]) for varname in varnames})
        else:
            raise IndexError('index not understood')

def _ifilter(filt,iterable):
    try:    
        return iterable[filt]
    except:
        if isinstance(filt, (int, float)):
            filt = [filt]
        return [val for i,val in enumerate(iterable) if i in filt]

def _maybe_hashmap(*collections):
    return [collection if isinstance(collection, Hashmap) else Hashmap(**collection)
            for collection in collections]

def _copy_hashmaps(*hashmaps):
    return [Hashmap(**{k:copy.deepcopy(v) for k,v in hashmap.items()})
            for hashmap in hashmaps]
