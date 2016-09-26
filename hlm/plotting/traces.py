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
        trace_kwargs = {'linewidth':.5, 'alpha':.80}
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

def corrplot(m, burn=0, thin=None,
             percentiles=[25,50,75], support=np.linspace(.001,1,num=1000),
             figure_kw=None, plot_kw=None, kde_kw=None):
    if figure_kw is None:
        figure_kw = {'figsize':(1.6*8,8), 'sharey':True}
    
    if plot_kw is None:
        plot_kw = [dict()]*len(percentiles)
    elif isinstance(plot_kw, dict):
        plot_kw = [plot_kw]*len(percentiles)
    elif isinstance(plot_kw, list):
        assert len(plot_kw)==len(percentiles)
    
    if kde_kw is None:
        kde_kw = [{'horizontal':True, 'shade':True}]*len(percentiles)
    elif isinstance(kde_kw, dict):
        kde_kw = [kde_kw]*len(percentiles)
    elif isinstance(kde_kw, list):
        assert len(kde_kw)==len(percentiles)
    
    corrfunc = m.state.correlation_function
    pwds = m.state.pwds
    if m.trace.n_chains > 1:
        raise
    phis = m.trace['Phi', burn::thin]
    f,ax = plt.subplots(1,2, **figure_kw)
    support = np.linspace(.001,1,num=1000)
    ptiles = [[np.percentile(corrfunc(r, pwds).flatten(), ptile) 
               for r in support] for ptile in percentiles]
    for i, ptile in enumerate(ptiles):
        ax[0].plot(support*m.state.max_dist, ptile, **plot_kw[i])
        sns.kdeplot(ptile, ax=ax[1], **kde_kw[i])
        ax[1].set_title(str(percentiles[i]))
    return f,ax
