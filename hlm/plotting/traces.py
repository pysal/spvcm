import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_trace(model, burn=0, thin=None, varnames=None, trace=None, 
               kde_kwargs={}, trace_kwargs={}, figure_kwargs={}):
    """
    Make a trace plot paired with a distributional plot.

    Arguments
    -----------
    trace   :   namespace
                a namespace whose variables are contained in varnames
    burn    :   int
                the number of iterations to discard from the front of the trace
    thin    :   int
                the number of iterations to discard between iterations
    varnames :  str or list
                name or list of names to plot.
    kde_kwargs : dictionary
                 dictionary of aesthetic arguments for the kde plot
    trace_kwargs : dictionary
                   dictinoary of aesthetic arguments for the traceplot

    Returns
    -------
    figure, axis tuple, where axis is (len(varnames), 2)
    """
    if model is None:
        if trace is None:
            raise Exception('Neither model nor trace provided.')
    else:
        trace = model.trace
    if varnames is None:
        varnames = trace.varnames
    elif isinstance(varnames, str):
        varnames = [varnames]
    if figure_kwargs == dict():
        figure_kwargs = {'figsize':(1.6*6, 12), 'sharey':'row'}
    if kde_kwargs == dict():
        kde_kwargs = {'shade':True, 'vertical':True}
    if trace_kwargs == dict():
        trace_kwargs = {'linewidth':.5} 
    fig, ax = plt.subplots(len(varnames), 2, **figure_kwargs)
    for chain_i, chain in enumerate(trace.chains):
        for i, param_name in enumerate(varnames):
            this_param = np.asarray(trace[chain_i,param_name,burn::thin])
            if len(this_param.shape) == 3:
                n,a,b = this_param.shape
                this_param = this_param.reshape(n,a*b)
            if len(this_param.shape) == 2:
                if this_param.shape[-1] == 1:
                    sns.kdeplot(this_param.flatten(),
                                ax=ax[i,1], **kde_kwargs)
                else:
                    for param in this_param.T:
                        sns.kdeplot(param, ax=ax[i,1], **kde_kwargs)
            else:
                sns.kdeplot(this_param, ax=ax[i,1], **kde_kwargs)
            ax[i,0].plot(this_param, **trace_kwargs)
            ax[i,1].set_title(param_name)
    fig.tight_layout()
    return fig, ax

