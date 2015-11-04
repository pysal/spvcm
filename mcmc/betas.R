VV <- XTX/sigma2e[i-1] + invT0
#### We use Cholesky decomption to inverse the covariance matrix
vBetas <- chol2inv(chol(VV))
if(inherits(vBetas,"try-error")) vBetas <- solve(VV)

### Define A=I-rho*W
A <- update_A(A,ind=ind2,rho=rho[i-1],W=W)
Ay <- as.numeric(A%*%y)
ZUs <- as.numeric(Z%*%Us[i-1,])
mBetas <- vBetas%*%(tX%*%(Ay-ZUs)/sigma2e[i-1]+T0M0)

#### When the number of independent variables is large, drawing from multivariate 
#### distribution could be time-comsuming
#cholV <- t(chol(vBetas))
##### draw form p-dimensitional independent norm distribution
betas <- rmvnorm(1, mean=mBetas, sigma=vBetas)
#betas <- rnorm(p)
#Betas[i,] <- betas <- mBetas + cholV%*%betas
Betas[i,] <- betas
