################################# Giddy Gibbs integration and inverse sampling for lambda
uu <- crossprod(us)
uMu <- as.numeric(t(us) %*% M %*% us)
Mu <- as.numeric(M %*% us)
uMMu <- crossprod(Mu)

lambda_exist <- detvalM[,1]
nlambda <- length(lambda_exist)
log_detlambda <- detvalM[,2]
iota <- rep(1,times=nlambda)


S_lambda <- uu*iota - 2*lambda_exist*uMu + lambda_exist^2*uMMu

##### Calculate the Log-density
log_den <- log_detlambda - S_lambda/(2*sigma2u[i-1]) 
adj <- max(log_den)
log_den <- log_den-adj

##### the density
den <- exp(log_den)
###### the interval separating lambda is h=0.001
h <- lambda_exist[2] - lambda_exist[1]

##### Integration to calculate the normalized constant
##### using the  trapezoid rule

ISUM <- sum(den)
norm_den <- den/ISUM
##### cumulative density
cumu_den <- cumsum(norm_den)
#
###### Inverse sampling
#rnd <- rval*sum(norm_den)
#ind <- which(cumu_den <= rnd)
#idraw <- max(ind)
#if(idraw > 0 & idraw < nlambda) lambda_draw <- lambda_exist[idraw]
