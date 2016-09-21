import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_trace(model, burn=0, varnames=None, trace=None, kde_kwargs={}, trace_kwargs={}):
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
    fig, ax = plt.subplots(len(varnames), 2, figsize=(1.6*6, 12), sharey='row')
    for i, parameter in enumerate(varnames):
        this_param = np.asarray(trace[parameter])
        if len(this_param.shape) == 3:
            a, b, _ = this_param.shape
            this_param = this_param.reshape(a,b)

        if len(this_param.shape) == 2:
            if this_param.shape[-1] == 1:
                sns.kdeplot(this_param.flatten()[burn:], 
                            shade=True, vertical=True, ax=ax[i,1],
                            **kde_kwargs)
            else:
                for param in this_param.T:
                    sns.kdeplot(param[burn:], shade=True, 
                                vertical=True, ax=ax[i,1],
                                **kde_kwargs)
        else:
            sns.kdeplot(this_param[burn:],
                        shade=True, vertical=True, ax=ax[i,1],
                        **kde_kwargs)
        ax[i,0].plot(this_param[burn:], linewidth=.5, **trace_kwargs)
        ax[i,1].set_title(parameter)
    fig.tight_layout()
    return fig, ax
