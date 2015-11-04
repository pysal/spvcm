#######################################################
####Starting the MCMC SET UP              #############
#######################################################

#### Uniform distribution for the spatial auto-regressive parameters rho, and lambda
 

n <- dim(X)[1]
p <- dim(X)[2]


################ Prior distribution specifications

### For Betas

M0 <- rep(0,times=p)
T0 <- diag(100,p)

### For sigma2e and sigma2u 
### completely non-informative priors

c0=d0=a0=b0=0.01

################ Store MCMC results
Betas <- matrix(0,nrow=Nsim,ncol=p)
Us <- matrix(0,nrow=Nsim,ncol=Utotal)
sigma2e <- rep(0,times=Nsim)
sigma2u <- rep(0,times=Nsim)
rho <- rep(0,times=Nsim)
lambda <- rep(0,times=Nsim)

#### initial values for model parameters (better initial values will be that from a classic multilevel model)
sigma2e[1] <- 2
sigma2u[1] <- 2
rho[1] <- 0.5
lambda[1] <- 0.5

################ Fixed posterior hyper-parameters, 
################ the shape parameter in the Inverse Gamma distribution

ce <- n/2 + c0
au <- Utotal/2 + a0


################ Fixed matrix manipulations during the MCMC loops

XTX <- crossprod(X)
invT0 <- solve(T0)
T0M0 <- invT0%*%M0
tX <- t(X)
tZ <- t(Z)
################# some fixed values when updating rho

beta0 <- solve(X,y)
e0 <- y-X%*%beta0
e0e0 <- crossprod(e0)

Wy <- as.numeric(W%*%y)
betad <- solve(X,Wy)
ed <- Wy-X%*%betad
eded <- crossprod(ed)

e0ed <- crossprod(e0,ed)


################# initialise A

if (class(W) == "dgCMatrix") {
	I <- sparseMatrix(i=1:n,j=1:n,x=Inf)
	A <- I - rho[1]*W
	ind <- which(is.infinite(A@x))
	ind2 <- which(!is.infinite(A@x))
	A@x[ind] <- 1
}

################# initialise B

if (class(M) == "dgCMatrix") {
	I <- sparseMatrix(i=1:Utotal,j=1:Utotal,x=Inf)
	B <- I - lambda[1]*M
	indB <- which(is.infinite(B@x))
	ind2B <- which(!is.infinite(B@x))
	B@x[indB] <- 1
}
