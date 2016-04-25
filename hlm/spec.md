# Specifying Models using Gibbs Samplers

The following details how to work with the Gibbs sampler framework deployed
here. 

At its core, a "Model" is a Gibbs sampler configured to sample from a set of one
or more conditional posterior distributions. A fully implemented model is:
1. A `User class` that
    - parses user input in any supported form (pandas + patsy, Numpy)
    - verifies that the input conforms to conditions (no NaN, conformal y,X, etc)
    - dispatches to the correct `Base class`
2. A `Base class` that
    - stores all of the required input variables in a shared `Namespace`. 
    - initializes a set of at least one conditional posterior distribution
      with correct hyperparameters and starting values
    - initializes the `Gibbs` framework, overriding default step/cycle behavior
      when necessary
    - computes some number of cycles by default
3. A set of Conditional Posterior Distributions that:
    - *may* (not must) be subclasses of `AbstractSampler`
    - correctly pull required values *from* the shared state
    - correctly assign derived quantities *to* the shared state
    - compute a new parameter from the required values when called.

## Defining a sampler

A Gibbs sampler is composed of a list of $p$ distributions $P(.|\mathcal{S_t})$,
where $\mathcal{S_t}$ is the internal state of the sampler.

Each $P(.|S_t)$ computes the value of its attached parameter using the current
values of the parameters stored in the state. For example, if you have a model
with three parameters, $\alpha$, $\beta$, $\theta$, where 

$$\alpha \sim \mathcal{N}(.5 * \beta, \theta)$$
$$\beta \sim \mathcal{G}(\alpha, \theta)$$
$$\theta \sim \mathcal{IG}(\alpha/2, \beta)$$

then the Gibbs sampler describing that setup would use the conditional distributions:
$$ P(\alpha_{t+1} | \beta_t, \theta_t)$$
$$ P(\beta_{t+1} | \alpha_{t+1}, \theta_t)$$
$$ P(\theta_{t+1} | \alpha_{t+1}, \beta_{t+1})$$

The ordered set of these distributions is called $\mathcal{P}$.  Also, let us
say there are $p$ elements in $\mathcal{P}$. The order of elements in
$\mathcal{P}$ is arbitrary, in theory, but any implementation will likely
structure the order to help memoize certain expensive computations. 

#### Moving around

With this, we can define the following terms:
- cycles : $t$, the number times all of the conditional posteriors $P_i \in \mathcal{P}$ have been executed.
- position : the index of the current conditional posterior $P_i$ in $\mathcal{P}$
- steps : the number of cycles times the number of positions in a cycle, that is, the total number of draws from any $P_i$. It is always equal to $cycles * p + position$.

Now we can define a few types of cycles and give some meaning to the parameter sets that come out of the process:
- full cycle: a cycle where the starting and ending positions are the same
- partial cycle: a cycle where the starting and ending positions are not the same
- perfect cycle: a cycle where the ending position is zero
- draw : a number or set of numbers drawn from $P_i$, where $i = position$. A draw where only one $P_i$ is used is called a 'step'.
- sample : some number, $d$, of draws.

Any type of cycle is also a type of sample. Thus, a full perfect sample is one drawn from a cycle that starts and ends at 0. The easiest way to get $s$ samples from the Gibbs sampler is to simply make $s$ full perfect cycles and record the results. 

#### Using the sampler

Let's call our sampler `G`. At its start, it has a position of $0$, and has
taken $0$ steps. If the user runs `G.step()`. it will compute a draw of $P_0$
and increment the position by one. If we have more than one $P_i \in
\mathcal{P}$, it will now take an additional partial perfect cycle to get a full
sample. 


By default the most recent draws are stored by default in the sampler state,
`G._state`. The set of most recent draws, either in the current cycle or in the
previous cycle, is called the *front* sample, the front of the sampler, or the
front parameters.  That is, the front sample constitutes the most recent full
imperfect sample. If you recall the discussion above, this is the set of
parameters needed by each conditional posterior. 

It's useful to contrast the *front* sample with the *current* sample. The
*current* sample reflects the draws from the current full perfect cycle. Thus,
it contains draws that have already been taken from $P_i$ where $i < position$,
and `None` for draws in the current full perfect cycle that have not yet been
computed.  Likewise, the *previous* parameters are a full perfect sample
computed during the last full perfect cycle. It never has `None' values, but
there may be more recent valid draws.  

Again, this terminology is important, since the Gibbs sampler uses the *front*
parameters. Only the *front* parameters, some of which are current and some of
which are previous, are contained within the `_state` of the sampler, which
contains all of the parameters, constants, and derived quantities needed to
operate the sampler *in any position*.

The `_state` is actually subclass of a dictionary, called a Namespace. You can
do anything with it that you can do with a dictionary. But, it makes its
elements available in dot notation as well. So, the following are both valid
ways to retrieve something from the sampler state:
```python
G._state.Betas
G._state['Betas']
```
If you try to store something that would override a dict property or attribute (like `update` or `clear`), a `TypeError` is raised. 

When the sampler steps forward (its position increments, one draw is computed),
the old draw of the parameter is stored in the `trace` Namespace and the new
draw replaces the old draw. In addition, if a derived quantity is computed in
one step that is used in later steps during the current cycle, that quantity is
held in the `_state`. Thus, you could "rerun" the sampler by resetting the
values held in state to the values held at some cycle in the trace.

#### Writing new model

New models are created by adding a new folder in this directory. The name should
be all lowercase, describe either the person who first derived the model (i.e.
the dong_harris model) or should attempt to describe the model form (i.e. hsar,
sar, car). 

A full model consists of:
1. a Base class/User class pair in `model.py`
2. a set of conditional posterior distributions to sample from in `samplers.py`. 

##### User classes

User classes can accept arbitrary structured input (Numpy arrays, pandas
dataframes, patsy formulas), and must parse input, check for consistency 

##### Base classes

Its user class/base class pair should live in ./{NAME}/model.py. The base class
should only take:
1. Numpy arrays for data (like `y,x,Wmatrix, Delta`)
2. starting values for parameters (done using `**tuning` in HSAR)

The base class needs to add all the needed variables to `self._state,` either via
using `self._state.update({dict_from_computations})` or by referring directly to
variables in `self._state`, like `self._state.X`.  This can be made more
convenient for short, one-off assignments by  abbreviating `self._state` to something very short in the early part of the init,  like `s`, and do all the required setup computations directly using dot notation:  
    ```python 
    s.XtX = np.dot(s.X.T, s.X)
    ```
2. write a separate method, like `_setup_data(self)` that makes a large amount of
   local variables. Then, at the end of it, return the locals() dictionary::
   ```python
   def _setup_data(self):
        s = self._state
        a2 = s.a**2
        a2ta = np.dot(a2.T, s.a)
        b = s.beta.T / a2ta
        return locals()
   ```
   in the context where this is called, you can then update `self._state` with
   your new variables if they will need to be shared among samplers:
   ```python
   results = self._setup_data()
   self._state.update(results)
   ```
   this will add all of the assignments made to local variables in `_setup_data(self)`
   to the sampler state. You could also return a subset of the local variables
   and use this strategy. 

This is done so that the individual conditional posteriors have a collection of
shared state, the model state, that they can use to make the many local
variables they compute persistent. They are described in the section called
**Conditional Posterior Samplers**. 

Once the samplers are constructed, they are passed to the `Gibbs`  init.  This
function automatically sets up the methods and attributes necessary to correctly
step through the conditional posteriors and store the results properly. Any
model that uses Gibbs sampling to estimate the conditional posterior
distributions should inherit from this class and call its `__init__` function
when done setting up the state.

If a model needs to change the default way:
- results are logged
- position is increased
- cycles are increased
from within the init function, then the changes must be made after this. The
other way this can be done is to override the different `cycle`, `sample`,
`step`, or `position` methods. It can also be done by  defining a custom `Gibbs`
subclass that the model inherits from. 

Finally, the base class calls the `sample` method, defined for any `Gibbs`
sampler. This takes a number of steps and cycles as keyword arguments and takes
a corresponding number of draws. As stated above, the total number of
draws/increments to the position of the sampler is $steps + p * cycles$. So, if
a sampler has 3 distributions and the user requests a sample of 2 steps and 3
cycles, then the user will take 11 draws overall, and will get 3 full samples
and one partial sample. If the user wishes to just setup the model and not
sample, then they should pass cycles=0 to the base class or user class. 

#### Conditional Posterior Samplers

After the base class sets up the data, it must set up the conditional posterior
samplers. These are simple classes with two methods: an `__init__` and a
`_cpost`. The skeleton of the class is defined in abstracts.AbstractSampler.
Each sampler is attached to the state that the entire model keeps track, the
base class's `self._state`. In this way, each conditional posterior is sharing
information, and doesn't have to recompute every variable it needs from the
current parameters. The `__init__` also assigns an initial parameter value to the
sampler. 

The `_cpost` function takes no arguments and should be a computational
implementation of $P_i(\theta | \mathcal{S})$, where $\mathcal{S}$ is the model
state. Since each of the samplers' state simply points to the overall model
state as done in the `__init__` function, any change that a sampler makes
changes the state of the whole model. The `_cpost` function should both assign
the new parameter value to the state **and** return it. So, if your parameter
value were theta, the last two lines of the `_cpost` should read:
    ```python
    st.theta = new_theta
    return st.theta
    ```
Right now, the return value is thrown away, but in the future it may be useful
to separate the assignment to state and the return value of the conditional
posterior
