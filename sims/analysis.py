import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_context('talk')
mpl.rcParams['figure.figsize'] = 9*1.6,9

def mkplots(df, truerho, truelam, prefix=None):
    """
    make plots of the trace runs
    """

    truebetas = [3.,6.,0.,2.]
    truegammas = [5.,-4.]
    truesigma_e = [1]
    truesigma_u = [.5]
    truerho = [truerho]
    truelam = [truelam]

    varplot(df, 'beta', truebetas,prefix)
    varplot(df, 'gamma', truegammas,prefix)
    varplot(df, 'rho', truerho,prefix)
    varplot(df, 'lambda', truelam,prefix)
    varplot(df, 'sigma_e', truesigma_e,prefix)
    varplot(df, 'sigma_u', truesigma_u,prefix)


def varplot(df, name,truth, prefix=None):
    plt.figure()
    for i,col in enumerate([x for x in df.columns if x.startswith(name)]):
        if col in ['rho', 'lambda', 'sigma_e', 'sigma_u']:
            lab = '$\\{}$'.format(col)
        else:
            lab = "$\\{}_{}$".format(col[:-1], col[-1])
        m = df[[col]].mean().values
        plt.plot(df[[col]], label= '{} = ${}$, $({})$'.format(lab, truth[i],
            m[0]))
    plt.title("Plot of $\\{}$".format(name))
    plt.legend()
    if prefix is None:
        filepath = name
    else:
        filepath = prefix + '_' + name + '.png'
    plt.savefig(filepath)
    plt.close('all')

if __name__ == '__main__':
    import os
    
    fnames = ['./results/' + x for x in os.listdir('./results')]
    parvals = [f.rstrip('.csv').replace('n','-').split('_') for f in os.listdir('./results')]
    parvals = [(float(r), float(l)) for r,l in parvals]
    columns = ['beta0','beta1','beta2','beta3','gamma0','gamma1','rho','lambda','sigma_e','sigma_u']

    for i, (fname, (r,l)) in enumerate(zip(fnames, parvals)):
        outpref = fname.rstrip('.csv')
        mkplots(pd.read_csv(fname, names=columns).drop(0), r, l, prefix=outpref)
