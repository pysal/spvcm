import copy
import types 

class Context(object):
    def __init__(self, clean=False):
        self._clean = clean
        self._knowns = {k:v for k,v in globals().items() 
                        if self._enter_cond(k,v)}
    
    def __enter__(self):
        return self

    def _enter_cond(self, k,v):
        """
        this needs to take a key,value pair,test it, and return a boolean if it
        should be included or excluded by the context manager. 
        """
        include = not k.startswith('_')
        include &= not isinstance(v, types.ModuleType)
        include &= not isinstance(v, type)
        return include

    def _changed_cond(self, k,v):
        """
        this needs to take a key,value pair, test it, and return a boolean if it
        should be considered "changed" during the context operation. 
        
        Typically, this should default to a comparison with globals() to detect
        if the value in self._knowns is equivalent its current value. It should
        also be a subset of _knowns. 
        """
        include = globals()[k] !=  v
        return include

    def _new_cond(self, k,v):
        """
        this needs to take a key,value pair, test it, and return a boolean if it
        should be considered "new" during the context operation. 

        Typically, this should be disjoint from "changed" and "new." 
        """
        include = k not in self._knowns
        include &= v is not self
        include &= not k.startswith('_')
        include &= not isinstance(v, types.ModuleType)
        include &= not isinstance(v, type)
        return include

    def __exit__(self, *args):
        self._new = {k:v for k,v in globals().items() 
                    if self._new_cond(k,v)}
        self._changed = {k:v for k,v in self._knowns.items()
                        if self._changed_cond(k,v)}
        self.diff = self._changed.copy()
        self.diff.update(self._new)
        if self._clean:
            self._cleanup()

    def _cleanup(self):
        """
        cleanup should replace all of the things that changed and delete the
        things that are new from the global namespace. 
        """
        for k in self.new:
            del globals()[k]
        for k in self.changed:
            globals().update({k:self._knowns[k]})


if __name__ == "__main__":
    outside = 3
    will_change = 4
    with Context() as con:
        will_change = 5
        new_var = 9

    with Context(clean=True) as con2:
        will_change = 1091
        waldo = 8
