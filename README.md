pyHSAR
========
*edit 11/4/2015: Whoops, realized I was pushing to my github version of this.
Work should be current now*.

This is @ljwolf's work on estimating an HSAR in python. 

I still need to:

<s>1. implement full conditionals for
    - lambda
    - rho
2. run the gibbs sampler</s> to validate the results.

I'll make a notebook explaining this, too. 

Validation, at this point, shows that some of the steps generate slightly
different results. I've already squashed a few bugs (and, consequently, cannot
wait for python 3's integrated [matrix multiplication operator,
`@`](https://www.python.org/dev/peps/pep-0465/) as well
as float division by default). 

Code
=====

- `trace.py` - mocks PyMC3 traces. deliberately designed to be easy to port to
PyMC3 when ready. 
- `samplers.py` - Gibbs sampler & sampling steps. The `Gibbs` class would be akin
to the PyMC3 `Metropolis`, `Slice`, or `NUTS` samplers. I might eventually push
this back upstream.

In esence, every `Gibbs` object has three things attached to it. 1 is a list of
the names of the parameters. 2 is a `Trace` object that stores data about the
sample. 3 is a list of full conditional posterior samplers. 1 & 3 could get
consolidated into a single property using an ordered dict, but I don't really
like that solution right now. Might change my mind later. 

- `setup.py` - builds the data and sets up the sampler. This is problem
  specific. You can configure the individual samplers to expect certain things
  from the trace using their "required" list. Right now, everything in the setup
  function gets passed down into the trace, so the trace has all of the objects
  it needs to run a sample. This allows us to flexibly define constants we need
  in the gibbs run, but also make it flexible for other configurations of the
  sampler. things are exported/expected using the required & exports lists.

Data
=====

- `test.csv` - taken from the dropbox folder `code/dong_harris_hsar`
- `w_lower.mtx` - taken from the dropbox folder above
- `w_upper.mtx` - taken from the dropbox folder above
`
