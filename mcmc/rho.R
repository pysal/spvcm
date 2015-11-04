################################# Giddy Gibbs integration and inverse sampling for rho
draw_rho <- function(detval,e0e0,eded,eueu,e0ed,e0eu,edeu,sig,rval) {
	rho_exist <- detval[,1]
	nrho <- length(rho_exist)
	log_detrho <- detval[,2]
	iota <- rep(1,times=nrho)
	#####Calculate Log-S(e(rho*)) given both rho* and u*
	S_rho <- e0e0*iota + rho_exist^2*eded + eueu - 2*rho_exist*e0ed - 2*e0eu + 2*rho_exist*edeu 

	##### Calculate the Log-density
	log_den <- log_detrho - S_rho/(2*sig) 
	adj <- max(log_den)
	log_den <- log_den-adj

	##### the density
	den <- exp(log_den)
	##### the interval separating rho is h=0.001
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
	if(idraw > 0 & idraw < nrho) rho_draw <- rho_exist[idraw]
}
betau <- solve(X,Zu)
eu <- Zu-X%*%betau
eueu <- crossprod(eu)
e0eu <- crossprod(e0,eu)
edeu <- crossprod(ed,eu)

#### Using the draw_rho function to update rho
rho[i] <- draw_rho(detval=detval,e0e0=e0e0,eded=eded,eueu=eueu,e0ed=e0ed,e0eu=e0eu,edeu=edeu,sig=sigma2e[i], rval)
