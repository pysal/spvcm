################################# Giddy Gibbs integration and inverse sampling for lambda
draw_lambda <- function(detval,uu,uMu,uMMu,sig) {
	lambda_exist <- detval[,1]
	nlambda <- length(lambda_exist)
	log_detlambda <- detval[,2]
	iota <- rep(1,times=nlambda)

	
	S_lambda <- uu*iota - 2*lambda_exist*uMu + lambda_exist^2*uMMu

	##### Calculate the Log-density
	log_den <- log_detlambda - S_lambda/(2*sig) 
	adj <- max(log_den)
	log_den <- log_den-adj

	##### the density
	den <- exp(log_den)
	##### the interval separating lambda is h=0.001
	h <- 0.001

	##### Integration to calculate the normalized constant
	##### using the  trapezoid rule

	ISUM <- h*(den[1]/2 + sum(den[2:1890]) + den[1891]/2)
	norm_den <- den/ISUM
	##### cumulative density
	cumu_den <- cumsum(norm_den)

	##### Inverse sampling
	rnd <- rval*sum(norm_den)
	ind <- which(cumu_den <= rnd)
	idraw <- max(ind)
	if(idraw > 0 & idraw < nlambda) lambda_draw <- lambda_exist[idraw]

	return(lambda_draw)
}

uu <- crossprod(us)
uMu <- as.numeric(t(us) %*% M %*% us)
Mu <- as.numeric(M %*% us)
uMMu <- crossprod(Mu)

lambda[i] <- draw_lambda(detval=detvalM,uu=uu,uMu=uMu,uMMu=uMMu,sig=sigma2u[i], rval)
