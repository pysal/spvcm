#### HSAR demo
#### The packages listed at the begining the function should be installed

library(MCMCpack) # using the inverse gamma generating function
library(mvtnorm)  # using multivariate normal generating function
library(spdep)    # optional
library(distr)	
library(spatialprobit) # calculate the vectorised log-determinants
library(Matrix)
#library(stringr)	

db <- read.csv('test.csv')
data <- db
W <- readMM('w_lower.mtx')
M <- readMM('w_upper.mtx')
# Note W and M are not of a specific sparse matrix of 'dgCMatrix' that are needed in the function

### order the data according to the "county"
### extract the number of counties(U) and individuals within each U
data <- data[order(data$county),] 
MM <- as.data.frame(table(data$county))
Utotal <- dim(MM)[1] # the number of counties in the data
Unum <- MM[,2] # the number of individuals that each county has
Uid <- rep(c(1:Utotal),Unum)# really just index each observations building linkage with the random effect

### Extract the random effect design matrix, Z, 919*85 for this data
n <- dim(data)[1]
Z <- matrix(0,nrow=n,ncol=Utotal)
for(i in 1:Utotal) {
  Z[Uid==i,i] <- 1
}
rm(i)

### change to a sparse matrix of "dgCMatrix"
Z <- as(Z,"dgCMatrix")

### Arguments used in the function
y <- data$y
X <- as.matrix(data$x)

### Need to provide log-determiant of the Jacobian terms of both the lower and higher levels
rmin <- -0.9
rmax <- 0.99
# convert spatial weights matrix to a required format 
# I really should apologise that I forgot this in the HSAR.R function.
W <- as(W,"dgCMatrix")
M <- as(M,"dgCMatrix")
tmp <- sar_lndet(ldetflag=2,W,rmin,rmax)
detval <- tmp$detval
tmp <- sar_lndet(ldetflag=2,M,rmin,rmax)
detvalM <- tmp$detval

### check for if spatial weight matrices W and M are valid

if (!inherits(W,"sparseMatrix") || any(diag(W) != 0)) {
	stop("W should be of sparse matrix form and diagonal elements should be 0s")	
}

if (!inherits(M,"sparseMatrix") || any(diag(M) != 0)) {
	stop("M should be of sparse matrix form and diagonal elements should be 0s")	
}

### change the dense random effect design matrix to a sparse one (optional)
Z <- as(Z,"dgCMatrix") 

start.time <- Sys.time()

### Some key functions in the estimation

### faster update of matrix A = I - rho*W for new values of rho
# @param A template matrix of (I - rho*W)
# @param ind indices to be replaced
# @param W spatial weight matrix with sparse matrix form
# @return (I - rho*W)

update_A <- function(A,ind,rho,W) {
	A@x[ind] <- (-rho*W)@x
	return(A)
}
###

### faster update of matrix B = I - lambda*M for new values of rho
# @param B template matrix of (I - lambda*M)
# @param ind indices to be replaced
# @param M spatial weight matrix with sparse matrix form
# @return (I - lambda*M)

update_B <- function(B,ind,lambda,M) {
	B@x[ind] <- (-lambda*M)@x
	return(B)
}
#######################################################
####Starting the MCMC SET UP              #############
#######################################################

Nsim = 1000 #preallocation

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

