# Exercise 1
# a
exercise_1_hilbert_a <- function(n) return( 1/ (outer(1:n, 1:n, `+`) - 1) )
exercise_1_a_result <- exercise_1_hilbert_a(10)
print(exercise_1_a_result)

#b Mathematically yes all of the Hilbert matricies are invertible as for any Hn
# all the vectors whose inner product was used for its construction 1, x .... x^n are linearly independent
# as they form the basis of Rn, which means that det(Hn) != 0


#c The issue with checking for the inverse of the hilbert matrix is that even for very small values of n,
# both its determinant as some of its eigenvalues become increasingly small, somewhere in the magnitute of <1e-10
# which basically means that they will be approximated to 0 and R would falsely conclude that the matrix is not invertible

exercise_1_is_invertible_c <- function(){
  n <- 0

  while(n < 10){
    n <- n + 1
    hilbert <- exercise_1_hilbert_a(n)
    
    tryCatch({inverse_hilbert <- qr.solve(hilbert)
    }, error = function(e){
      print(paste("At Hilbert matrix number", n, ' R thinks it is singular i.e. not-invertible'))
      return(FALSE)
    })
    
  }
  return(TRUE)
}
exercise_1_is_invertible_result <- exercise_1_is_invertible_c()


# Exercise 2
exercise_2 <- function(x_vector){
  b = c(25, 16, 26, 19, 21, 20)
  # The idea here is that the x-values will 1, x, x^2, x^3, x^4, x^5 will be the coefficients in the matrix A "x" will be
  # the coefficients a1,a2,a3,a4,a5 and b are the final values 25,16,26,19,21,20 and we will solve Ax = b
  
  # Intitiate an empty matrix
  A <- matrix(numeric(0), 6, 6)
    
  # Calculate the A matrix with each x value sequence as a row
  for (element in x_vector){
    powers <- seq(0,5)
    row_vector <- element ^ powers
    A <- rbind(A, row_vector)
  }
  
  # Drop the NULL values since we had to initiate them in making an empty matrix
  A <- A[complete.cases(A),]
  
  # Solve Ax = b
  coefficients_vector <- qr.solve(A, b)
  return(coefficients_vector)
}
exercise_2_result <- exercise_2(c(10,11,12,13,14,15))
print(exercise_2_result)


# Exercise 3

exercise_3_calculate_h <- function(){
  
  # Generate 15 uniformally distributed elements
  elements <- runif(15, 0, 1)
  
  # Initiate the X matrix
  X <- matrix(elements, 5 , 3)
  
  # Calculate H
  H <- qr.solve(t(X) %*% X)
  H <- X %*% H
  H <- H %*% t(X)
  return(list(H = H, X = X))
}
result <- exercise_3_calculate_h()
H <- result$H
X <- result$X

# a
exercise_3_a <- function(H){
  return(eigen(H))
}
eigen_list <- exercise_3_a(H)

# b
exercise_3_b <- function(H, eigen_list){
  trace_h <- sum(diag(H))
  eigen_sum <- sum(eigen_list$values)
  print(paste("Trace is", trace_h, "and the sum of the eigenvalues is", eigen_sum))
}
exercise_3_b(H, eigen_list)


# c
exercise_3_c <- function(H, eigen_list){
  print(paste("The determinant of H is", det(H), "the product of the eigen values is", prod(eigen_list$values)))
}
exercise_3_c(H, eigen_list)

# d
exercise_3_d <- function(H, X){
  print(X)
  print(eigen(H)$vectors)
  print(paste("It is", all.equal(X, eigen(H)$vector) , "that X is the same as the eigenvectors of H"))
  print(paste('The corresponding eigen values are'))
  print(eigen(H)$value)
}
exercise_3_d(H, X)


# Exercise 4
exercise_4 <- function(){
  hilbert <- exercise_1_hilbert_a(6)
  
  eigen_values <- eigen(hilbert)$value
  eigen_vectors <- eigen(hilbert)$vectors
  
  hilbert_inverse <- qr.solve(hilbert)
  eigen_values_inverse <- eigen(hilbert_inverse)$value
  print("The Eigen values of the Hilbert Matrix are")
  print(format(eigen_values, scientific = FALSE))
  
  print("The Eigen Values of the Inverse Hilbert Matrix are")
  print(format(eigen_values_inverse, scientific = FALSE))
}
exercise_4()
# The relationship is that the eigen values of the inverse are reciprocal to the eigen values of the Hilbert matrix 
# and that both matrixes have the same condition number



# Exercise 5
# a
exercise_5_a <- function(){
  P <- matrix(NA, 4, 4)
  values <- seq(1,4)/10
  
  for (i in 1:4){
    if (i == 1){
      P[i, ] <- values[i:4]
    }else{
    P[i,] <- c(values[i:4], values[1:(i-1) ])
    }
  }
  
  row_sums <- apply(P, 1, sum)
  
  print(row_sums)
  
  return(P)
}
 exercise_5_a()
# The output shows how all the row sums are 1


# b
exercise_5_b <- function(vector){
  
  P <- exercise_5_a()
  
  print(P)
  
  P_2 <- P %*% P
  
  print(P_2)
  
  P_3 <- P %*% P_2
  
  print(P_3)
  
  P_5 <- P_3 %*% P_2
  
  print(P_5)
  
  P_10 <- P_5 %*% P_5
  print(P_10)
  
}
exercise_5_b()


# The pattern is that the value inside each cell gets closer and closer to 1/nrow()

# C
# The x vector is nothing different than the eigenvector coresponding to the eigen value of 1
exercise_5_c <- function(){
  
  P <- exercise_5_a()
  
  # We find the eigenvector that has an eigen value of 1 and normalize it
  
  eig <- eigen(t(P))

  x <- eig$vectors[, which.min(abs(eig$values - 1))]
  x <- x / sum(x)
  
  print("The vector is")
  print(x)
  
  print("To verify P` %*% x = x")
  print(t(t(P) %*% x))
}
exercise_5_c()



# Exercise 8 
# The structure implies 
# L1x = b ; Bx + L2y = c, we solve L1x by forward substitution, after finding x we substitute into 
# Bx + L2y = c, now L2 is lower triangular so we solve by backward substitution L2y = c - Bx


# Exercise 27
exercise_27 <- function(polynomial_vector){
  matrix <- c()
  
  # This will keep trac where we add the 1s in the matrix
  itt <- 0 
  
  # We will create the companion matrix row by row
  # We loop over every element except the last one, which is 1 i.e. not included in the companion matrix
  for (coefficient in polynomial_vector[-length(polynomial_vector)]){
    # Each itteration the row is 0,0,0,0,0,0,0 or n-2 zeroes has a 1
    # whose position depends on the coefficient being referenced and the last cell is -element
    row <- numeric(length(polynomial_vector) - 2)
    
    # This adds the 1 on each row
    if ( itt != 0 ) row[itt] <- 1
    
    # This adds the -coefficient and finishes creating the row
    row <- c(row, -coefficient)
    
    # We append each vector to the matrix as a row
    matrix <- rbind(matrix, row)
    itt <- itt + 1
  }
  eigen_values <- eigen(matrix)$values
  return(eigen_values)
}
exercise_27_result <- exercise_27(c(24, -40, 35, -13, 1))
print("The eigen values are")
print(exercise_27_result)


# Exercise 29
exercise_29 <- function(A){
  
  output <- c()

  for (i in 1:ncol(A)){
    
    output <- c(output, A[, i])
  }
  return(output)
}
exercise_29_result <- exercise_29(matrix(1:100, 20 ,5))
exercise_29_result


# Exercise 30
exercise_30 <- function(A){
  output <- c()
  
  for (i in 1:ncol(A)){
    output <- c(output, A[i:nrow(A), i])
  }
  return(output)
}
exercise_30_result <- exercise_30(matrix(c(1,2,3,2,4,5,3,5,6), nrow = 3))
exercise_30_result