# Exercise 41
exercise_41_a <- function(n, m, rho){
  sigma <- matrix(c(1, rho, rho, 1), nrow= 2)
  
  eigen_vectors <- eigen(sigma)$vectors
  eigen_values <- diag(eigen(sigma)$values)
  
  Z <- matrix(rnorm( n* length(m)), nrow = n)

  X <- m + Z %*% t(eigen_vectors %*% sqrt(eigen_values))
  print(X)
  
}
exercise_41_a(1000, c(0,3), 0.7)


exercise_41_b <- function(n ,m, rho){
  
  sigma <- matrix(c(1, rho, rho, 1), nrow = 2)
  L <- chol(sigma)
  Z <- matrix(rnorm(n * length(m)),nrow = n)
  print(L)
  X <- Z %*% t(L) + matrix(m, nrow = n, ncol = length(m), byrow = TRUE)
  print(X)
}
exercise_41_b(1000, c(0,3), 0.7)


# Exercise 42
exercise_42 <- function(n, dimensions,rho){
  F <- rnorm(1)
  Z <- matrix(rnorm(dimensions * n), nrow = n, ncol = dimensions)
  X <- sqrt(rho) * F + sqrt(1 - rho) * Z
  print(X)
}
exercise_42(1000, 10, 0.44)


# Exercise 43
exercise_43 <- function( n, v, m, rho = 0.3){
  
  d <- length(m)
  
  sigma <- matrix(rho, nrow = d, ncol = d)
  diag(sigma) <- 1
  chol_sigma <- chol(sigma)
  

  Z <- matrix(rnorm(n * d), nrow =n, ncol = d) %*% chol_sigma
  
  W <- sqrt( rchisq(n, df = v)/v)
  T <- sweep(Z, 1, W, '/') + matrix(m, nrow =n, ncol = d, byrow = TRUE)
  
  for (i in 1:d){
    qqnorm(T[,i])
    qqline(T[,i])
  }
  
  empirical_mean <- colMeans(T)
  empirical_cov <- cov(T)
  
  print(empirical_mean)
  print(empirical_cov)
  
}
exercise_43(1000, 4, c(0,3,4))

# Exercise 47
exercise_47 <- function(){
  set.seed(19908)
  U <- runif(1000)
  mean_U <- mean(U)
  var_U <- var(U)
  sd_U <- sqrt(var_U)
  
  print(paste("True mean 0.5 Empirical mean", mean_U))
  print(paste("True var 0.8333 Empirical var", var_U))
  print(paste("True sd 0.2886 Empirical sd", sd_U))
  print(paste("Theoretically 60% of values have to be below 0.6, empirically it is", sum(U < 0.6)/length(U)))
  
}
exercise_47()


# Exercise 48
exercise_48 <- function(){
  U1 <- runif(10000)
  U2 <- runif(10000)
  
  U1_U2 <- U1 + U2
  print(paste('True expected value is 1, empirically it is', mean(U1_U2)))
  print(paste('True var(U1 + U2) is 0.1667, empirically it is', var(U1_U2)))
  print(paste('True var(U1) + var(U2) is 0.1667, empiricially it is', var(U1) + var(U2)))
  print(paste('P(U1 + U2) <= 1.5', sum(U1_U2 <= 1.5)/ length(U1_U2)))
  print(paste('P(sqrt(U1) + sqrt(U2)) <= 1.5', sum((sqrt(U1) + sqrt(U2)) <= 1.5)/ length(U1_U2)))
  
}
exercise_48()

# Exercise 49
exercise_49 <- function(){
  U1 <- runif(10000)
  U2 <- runif(10000)
  U3 <- runif(10000)
  U1_U2_U3 <- U1 + U2 + U3
  
  print(paste('E(U1 + U2 + U3) is', mean(U1_U2_U3)))
  print(paste('var(U1 + U2 + U3) is', var(U1_U2_U3)))
  print(paste('var(U1) + var(U2) + var(U3) is', var(U1) + var(U2) + var(U3)))
  print(paste('E(sqrt(U1 + U2 + U3)) is', mean(sqrt(U1_U2_U3))))
  print(paste('P(sqrt(U1) + sqrt(U2) + sqrt(U3)) >= 0.8 is', sum((sqrt(U1) + sqrt(U2) + sqrt(U3)) >= 0.8)/length(U1_U2_U3)))
}
exercise_49()


# Exercise 50
exercise_50 <- function(){
  output <- rbinom(n = 100, size = 20, prob = 0.5)
  mean_mark <- mean(output)
  var_mark <- sd(output)
  
  print(paste('Mean is', mean_mark, 'standard deviation is', var_mark))
  print(paste('The proportion of students with a mark higher than 30% is', sum(output >= 0.3 * 20) / length(output) * 100, '%'))
}
exercise_50()


# Exercise 51
exercise_51 <- function(){
  outcome <- rbinom(n = 10000, size = 20, prob = 0.3)
  
  print(paste("P(X <= 5) is", sum(outcome <= 5) / length(outcome)))
  print(paste("P(X = 5) is", sum(outcome == 5) / length(outcome)))
  print(paste('E(X) = ', mean(outcome)))
  print(paste('Var(X) = ', var(outcome)))
  print(paste('95th Percentile is', quantile(outcome, probs = c(0.95))))
  print(paste('99th Percentile is', quantile(outcome, probs = c(0.99))))
  print(paste('99.9999th Percentile is', quantile(outcome, probs = c(0.999999))))
  
}
exercise_51()


# Exercise 52
exercise_52 <- function(n, size, prob){
  cumbins <- pbinom(0 : (size - 1), size, prob)
  singlenumber <- function(){
    x <- runif(1)
    return(sum(x > cumbins))
  }
  output <- sapply(1:n, function(x) singlenumber())
}
exercise_52(100, 20, 0.3)

new_method <- function(n, size, prob){
  output <- sapply(n, function(x) exercise_52(x, size, prob))
  return(output)
}
print(system.time(new_method(c(1000, 10000, 100000), 10, 0.4)))


old_method <- function(n, size, prob){
  output <- sapply(n, function(x) rbinom(x, size, prob))
  return(output)
}
print(system.time(old_method(c(1000, 10000, 100000), 10, 0.4)))

# rbinom() is an order of magnitude faster (10 times)


# Exercise 53
exercise_53 <- function(n, size, prob) {
  
   singlenumber <- function(size, prob) {
     x <- runif(size)
     return(sum(x < prob))
     }
   output <- sapply(1:n, function(x) singlenumber(size, prob) )
   return(output)
}

old_method <- function(n, size, prob){
  result <- sapply(size, function(x) exercise_53(n, x, prob)) 
  return(result)
}
print(system.time(old_method(10000, c(10, 100, 1000), 0.4)))

# Exercise 54
exercise54 <- function(n, size, prob) {
  
   singlenumber <- function(size, prob) {
     k <- 0
     U <- runif(1)
     X <- numeric(size)
     while(k < size) {
       k <- k + 1
       
       if(U <= prob) {
         X[k] <- 1
         print(paste('U is', U, 'updated U is', U/ prob))
         U <- U / prob
         
         } else {
           X[k] <- 0
           
           print(paste("___U is", U, 'updated U is', (U- prob) / (1- prob)))
           U <- (U - prob) / (1 - prob)
           }
       }
     return(sum(X))
     }
   output <- sapply(1:n, function(x) singlenumber(size, prob))
}

exercise_54_a <- exercise54(100, 20, 0.4)
exercise_54_b <- exercise54(100, 500, 0.7)



exercise_54_c <- function(n, size, prob){
  
  singlenumber <- function(size, prob) {
    k <- 0
    U <- runif(1)
    X <- numeric(size)
    U_vector <- c(U)
    
    while(k < size) {
      k <- k + 1
      
      if(U <= prob) {
        X[k] <- 1
        U <- U / prob
        U_vector <- c(U_vector, U)
        
      } else {
        X[k] <- 0
        
        U <- (U - prob) / (1 - prob)
        U_vector <- c(U_vector, U)
        
      }
    }
    U_vector <- U_vector[U_vector < prob]
    U_vector <- U_vector / prob
    hist(U_vector, breaks = 40)
    return(sum(X))
  }
  output <- sapply(1:n, function(x) singlenumber(size, prob))
  
}
exercise_54_c(100, 20, 0.4)


#Exercise 55
exercise_55 <- function(){
  
  run_once <- function(){
    n <- 2
    while(TRUE){
      birtdays <- sample(1:365, n, replace = TRUE)
      
      sampled_birthdays <- sample(birtdays, 2, replace = FALSE)

      if(sampled_birthdays[1] == sampled_birthdays[2]){
        break
      }else{
        n <- n + 1
      }
    }
    return(n)
    }
  outcome <- sapply(1:100, function(x) run_once())
  hist(outcome, breaks = 30)
}
exercise_55()
