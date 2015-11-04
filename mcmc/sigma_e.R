################################# Gibbs sampler for updating sigma2e
Zu <- rep(us,Unum) #### Z%*%us
e <- Ay-Zu-Xb
de <- 0.5*crossprod(e)+d0
sigma2e[i] <- rinvgamma(1,shape=ce,scale=de)
