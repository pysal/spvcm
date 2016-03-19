pyHSAR
========

This is @ljwolf's work on estimating an HSAR in python. 

Broadly speaking, I've implemented

1. A Gibbs sampler class that is composed of individual single-step samplers and a trace object.
2. Single-step samplers that contain some `_cpost` full conditional posterior
   function to sample a new value given the trace.
3. The trace, which is essentially a container for the state of the sampler,
   which has
        - Stochastics (things being sampled)
        - Statics (invariants throughout sampling)
        - Derived (quantities derived during sampling)
4. a design-of-experiments setup in `./sims/dgp.py` and `./sims/mc.py` against
   which we can test sensitivity and sampler results. 

You construct the Gibbs sampler, which has all the requisite code for
organizing, constructing, and conducting a run. The Trace should just be the
private namespace inside of which results & computational memos are stored. Each
sampler implements a `_cpost` method, and a `sample()` public method that calls
the `_cpost()` method. 

Monte Carlo simulation work is going in `sims/`. Right now, I'm working on a
"small" and a "big" scenario, and I will (hopefully) run all of the MC tests
through R and Python implementations. 

Structure
==========

- `trace.py` - mocks PyMC3 traces. deliberately designed to be easy to port to
  PyMC3 when ready. 
- `samplers.py` - Gibbs sampler & sampling steps. The `Gibbs` class would be
  akin to the PyMC3 `Metropolis`, `Slice`, or `NUTS` samplers. I might
  eventually push this back upstream. In esence, every `Gibbs` object has three
  things attached to it. 1 is a list of the names of the parameters. 2 is a
  `Trace` object that stores data about the sample. 3 is a list of full
  conditional posterior samplers. 1 & 3 could get consolidated into a single
  property using an ordered dict, but I don't really like that solution right
  now. Might change my mind later. 
- `setup.py` - builds the data and sets up the sampler. This is problem
  specific. You can configure the individual samplers to expect certain things
  from the trace using their "required" list. Right now, everything in the setup
  function gets passed down into the trace, so the trace has all of the objects
  it needs to run a sample. This allows us to flexibly define constants we need
  in the gibbs run, but also make it flexible for other configurations of the
  sampler. things are exported/expected using the required & exports lists.
- `validate.py` - contains the code to validate the sampling runs. By validate,
  I mean compare intermediate computations against from R and Python. 
- `test.csv` - taken from the dropbox folder `code/dong_harris_hsar`
- `w_lower.mtx` - taken from the dropbox folder above
- `w_upper.mtx` - taken from the dropbox folder above
`
