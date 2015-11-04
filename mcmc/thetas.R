################################# Gibbs sampler for updating U. Now, us are spatially dependent.

### Define and update B=I-lambda*M

B <- update_B(B,ind=ind2B,lambda=lambda[i-1],M=M)

vU <- as.matrix(tZ%*%Z/sigma2e[i-1] + t(B)%*%B/sigma2u[i-1])
vU <- chol2inv(chol(vU))
if(inherits(vU,"try-error")) vU <- solve(vU)

Xb <- X%*%betas
mU <- vU%*%(as.numeric(tZ%*%(Ay - Xb))/sigma2e[i-1])

#### When the number of higher level units is large, drawing from multivariate 
#### distribution could be time-comsuming
#cholV <- t(chol(vU))
#
us <- rmvnorm(1, mean=mU, sigma=vU)
##### draw form J-dimensitional independent norm distribution
#us <- rnorm(Utotal)
#Us[i,] <- us <- mU + cholV%*%us
Us[i,] <- us
