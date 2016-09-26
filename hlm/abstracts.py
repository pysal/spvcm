import warnings
from datetime import datetime as dt
import numpy as np
import pysal as ps
import copy
import multiprocessing as mp
import sqlite3 as sql
from .sqlite import head_to_sql, start_sql
from .plotting.traces import plot_trace
import pandas as pd

######################
# SAMPLER MECHANISMS #
######################

class Sampler_Mixin(object):
    def __init__(self):
        pass

    def sample(self, n_samples, n_jobs=1):
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
        if n_jobs > 1:
           self._parallel_sample(n_samples, n_jobs)
           return
        if isinstance(self.state, list):
            self._parallel_sample(n_samples, n_jobs=len(self.state))
            return
        _start = dt.now()
        try:
            while n_samples > 0:
                if (self._verbose > 1) and (n_samples % 100 == 0):
                    print('{} Draws to go'.format(n_samples))
                self.draw()
                n_samples -= 1
        except KeyboardInterrupt:
            warnings.warn('Sampling interrupted, drew {} samples'.format(self.cycles))
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
        if self.cycles == 0:
            self._finalize_invariants()
        self._sample()
        self.cycles += 1
        for param in self.traced_params:
            self.trace.chains[0][param].append(self.state[param])
        if self.database is not None:
            head_to_sql(self, self._cur, self._cxn)
            for param in self.traced_params:
                self.trace[param] = [getattr(self.trace, param)[-1]]
    
    def _parallel_sample(self, n_samples, n_jobs):
        models = []
        for i in range(n_jobs):
            with warnings.catch_warnings() as w:
                warnings.simplefilter("ignore")
                if isinstance(self.state, list):
                    state = self.state[i] #keep state from previous parallel runs
                else:
                    state = self.state
                state['n_samples'] = 0
                state['n_jobs'] = 1
                new_model = type(self)(**state)
                if self.database is not None:
                    new_model.database = model.database + str(i)
                self._fuzz_starting_values(new_model.state)
            models.append(new_model)
        n_samples = [n_samples] * n_jobs
        seed = np.random.randint(0,10000, size=n_jobs).tolist()
        P = mp.Pool(n_jobs)
        results = P.map(_reflexive_sample, zip(models, n_samples, seed))
        P.close()
        if self.cycles > 0:
            new_traces = []
            for i, model in enumerate(results):
                new_traces.append(Hashmap(**{k:param + model.trace.chains[0][k] 
                                             for k, param in self.trace.chains[i].items()}))
            new_trace = Trace(*new_traces)
        else:
            new_trace = Trace(*[model.trace.chains[0] for model in results])
        self.trace = new_trace
        self.state = [model.state for model in results]
        self.cycles += n_samples[0]
    
    def _fuzz_starting_values(self, state):
        pass

    @property
    def database(self):
        return getattr(self, '_db', None)
        
    @database.setter
    def database(self, filename):
        self._cxn, self._cur = start_sql(self)
        self._db = filename

def _reflexive_sample(tup):
    """
    a helper function sample a bunch of models in parallel.
    
    Tuple must be:
    
    model : model object
    n_samples : int number of samples
    seed : seed to use for the sampler
    """
    model, n_samples, seed = tup
    np.random.seed(seed)
    model.sample(n_samples, n_jobs=1)
    return model

#######################
# MAPS AND CONTAINERS #
#######################

class Hashmap(dict):
    """
    A dictionary with dot access on attributes
    """
    def __init__(self, **kw):
        super(Hashmap, self).__init__(**kw)
        if kw != dict():
            for k in kw:
                self[k] = kw[k]

    def __getattr__(self, attr):
        try:
            r = self[attr]
        except KeyError:
            try:
                r = getattr(super(Hashmap, self), attr)
            except AttributeError:
                raise AttributeError("'{}' object has no attribute '{}'"
                                     .format(self.__class__, attr))
        retu

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Hashmap, self).__setitem__(key,value)
        self.__dict__.update({key:value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Hashmap, self).__delitem__(key)
        del self.__dict__[key]
        
class Trace(object):
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
        if isinstance(key, str): #user wants a name from the traces
            if self.n_chains  > 1:
                return np.asarray([chain[key] for chain in self.chains])
            else:
                return np.asarray(self.chains[0][key])
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
                        return np.asarray([_ifilter(tail, chain[head]) for chain in self.chains])
                    else:
                        return np.asarray(_ifilter(tail, self.chains[0][head]))
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
                        return np.asarray([chain[tail] for chain in target_chains])
                    elif len(target_chains) == 1:
                        return np.asarray(target_chains[0][tail])
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
                        return np.asarray([_ifilter(iters, chain[varnames[0]]) for chain in chains])
                    else:
                        return np.asarray(_ifilter(iters, chains[0][varnames[0]]))
                else:
                    if nchains > 1:
                        return [Hashmap(**{varname:_ifilter(iters, chain[varname])
                                        for varname in varnames})
                                for chain in chains]
                    else:
                        return Hashmap(**{varname:_ifilter(iters, chains[0][varname]) for varname in varnames})
        else:
            raise IndexError('index not understood')
            
    
    def to_df(self):
        dfs = []
        for chain in self.chains:
            df = pd.DataFrame().from_records(dict(**chain))
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
            dfs.append(df)
        if len(dfs) == 1:
            return dfs[0]
        else:
            return dfs

    def to_csv(self, filename, *pandas_args, **pandas_kwargs):
        """
        Write trace out to file. If there are multiple traces, this will write
        them each out to 'filename_number.csv', where number is the number of the trace.
        """
        dfs = self.to_df()
        if isinstance(dfs, list):
            name, ext = os.path.splitext(filename)
            for i, df in enumerate(dfs):
                df.to_csv(name '_' + str(i) + '.' + ext, *pandas_args, **pandas_kwargs)
        else:
            dfs.to_csv(filename, *pandas_args, **pandas_kwargs)

    def plot(trace, burn=0, thin=None, varnames=None,
             kde_kwargs={}, trace_kwargs={}, figure_kwargs={}):
        f, ax = plot_trace(model=None, trace=trace, burn=burn, 
                           thin=thin, varnames=varnames, 
                      kde_kwargs=kde_kwargs, trace_kwargs=trace_kwargs,
                      figure_kwargs=figure_kwargs)
        return f,ax

####################
# HELPER FUNCTIONS #
####################

def _ifilter(filt,iterable):
    try:
        return iterable[filt]
    except:
        if isinstance(filt, (int, float)):
            filt = [filt]
        return [val for i,val in enumerate(iterable) if i in filt]

def _maybe_hashmap(*collections):
    out = []
    for collection in collections:
        if isinstance(collection, Hashmap):
            out.append(collection)
        else:
            out.append(Hashmap(**collection))
    return out

def _copy_hashmaps(*hashmaps):
    return [Hashmap(**{k:copy.deepcopy(v) for k,v in hashmap.items()})
            for hashmap in hashmaps]
