################################# Gibbs sampler for updating sigma2u
Bus <- as.numeric(B%*%t(us))
bu <- crossprod(Bus)/2 + b0
sigma2u[i] <- rinvgamma(1,shape=au,scale=bu)
