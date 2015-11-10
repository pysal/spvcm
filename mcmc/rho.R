################################# Giddy Gibbs integration and inverse sampling for rho
betau <- solve(X,Zu)
eu <- Zu-X%*%betau
eueu <- crossprod(eu)
e0eu <- crossprod(e0,eu)
edeu <- crossprod(ed,eu)

rho_exist <- detval[,1]
nrho <- length(rho_exist)
log_detrho <- detval[,2]
iota <- rep(1,times=nrho)
#####Calculate Log-S(e(rho*)) given both rho* and u*
S_rho <- e0e0*iota + rho_exist^2*eded + eueu - 2*rho_exist*e0ed - 2*e0eu + 2*rho_exist*edeu 

##### Calculate the Log-density
log_den <- log_detrho - S_rho/(2*sigma2e[i-1]) 
adj <- max(log_den)
log_den <- log_den-adj

##### the density
den <- exp(log_den)
##### the interval separating rho is h=0.001
h <- rho_exist[2] - rho_exist[1]

##### Integration to calculate the normalized constant
##### using the  trapezoid rule

norm_den <- den/sum(den)
##### cumulative density
cumu_den <- cumsum(norm_den)
#
###### Inverse sampling
#rnd <- rval*sum(norm_den)
#ind <- which(cumu_den <= rnd)
#idraw <- max(ind)
#if(idraw > 0 & idraw < nrho) rho_draw <- rho_exist[idraw]
