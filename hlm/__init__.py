from . import both_levels as both #okay, because it's not * all the way down :)
from . import upper_level as upper
from . import lower_level as lower
from . import svcp as svcp
from . import hierarchical as hier
from .plotting.traces import plot_trace

del (both_levels, upper_level, lower_level, svcp, hierarchical, 
     trace, plotting, utils, verify, steps, abstracts)
