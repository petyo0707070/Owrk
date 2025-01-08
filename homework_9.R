# Exercise 60

exercise_60 <- function(){
  a <- 2
  b <- 2
  u_values <- runif(100, min =0, max = 1)
  x_values <- b/((1-u_values)^(1/a))
  hist(x_values, probability = TRUE, main = "Pareto(2,2) Density")
  
  # Keep in mind Pareto Density(2,2) = 8/(x^3)
  
  sequence_x_values <- seq(min(x_values), max(x_values), length.out = 100)
  
  density_curve <- 8/(sequence_x_values ^ 3)
  lines(density_curve, lwd = 2, col = 'blue')
}

exercise_60()


# Exercise 62
exercise_62 <- function(){
  mu <- runif(1, min = - 1, max = 1)
  e <- runif(1, min = -1, max = 0 - 1e-8)
  sigma <- runif(1, min = 0 + 1e-8, max = 1)
  u <- runif(1000, min = 0, max = 1)
  
  x_values <- mu + sigma/e * ((1-u)^(-e) - 1)
  hist(x_values, probability = TRUE, main = "Random Pareto Distribution")
}
exercise_62()


# Exercise 63
# The idea is that the cummilative probability function is the literal sum of probabilities up to a given x,
# This means F(x = 0) = 0.1, F(x = 1) = 0.3, F(x = 2) = 0.5, F(x = 3) = 0.7, F(x = 4) = 1
# The inverse of this cfd is min{x: F(x) >= p}, hence F_inv = 0 if 0 <= p < 0.1
# 1 if 0.1 <= p < 0.3, 2 if 0.3 <= p < 0.5, 3 if 0.5 <= p < 0.7, 4 if 0.7 <= p <= 1

exercise_63 <- function(){
  
  probability_values <- c(0.1, 0.3, 0.5, 0.7, 1)
  f_values <- seq(0,4)
  
  helper_function <- function(x){
      return(f_values[which(x <= probability_values)][1])
  }
  u <- runif(1000)
  
  outcome <- sapply(u, helper_function)
  print("Results from runif()")
  print(table(outcome))
  
  u_sample <- sample(seq(0,1,length.out = 1000), 1000)
  outcome_sample <- sapply(u_sample, helper_function)
  
  print("Results from sample()")
  print(table(outcome_sample))
  
}
exercise_63()
# The outcome from runif() is not precise there is deviation up to +/- 10 the theoretical outcome
# The outcome from sample() is accurate and on point