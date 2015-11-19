x0_CONST = np.random.random()*20 - 10
x0 = np.ones(N) * x0_CONST #constant

x1 = np.random.random(size=N)*10 - 5 #continuous x \in [-10,10]

x2 = np.random.random(size=N)*6 - 3 #continuous x \in [-3,3] won't be significant

x3 = np.random.randint(3, size=N) - 1 #balanced categorical x \in {-1,0,1}

X = np.vstack((x0,x1,x2,x3)).T
N2,p = X.shape #3 covariates + constant

z1 = np.random.random(size=J)*8 -4 #continuous x \in [-8,8]

Z = z1.T
