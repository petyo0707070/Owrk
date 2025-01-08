# Exercise 18
exercise_18 <- function(A){
  # The 2-norm of a matrix is the largest singular value
  # Singular values are found by taking the square root of the eigen values of A %*% t(A)
  svd_matrix <- svd(A)
  
  return(max(svd_matrix$d))
  
}
exercise_18_result <- exercise_18(matrix(seq(1,4), 2, 2))
print(paste("The 2-norm of A is", exercise_18_result))


# Exercise 19
  # The idea here is that the 2-norm was the largest singular value of A (max in V of the SVD decomposition)
  # k(A) = 2-norm(A) * 2-norm(A_inverse), we know that norm(A) = max(singular value), while the
  # the norm(A_inverse) is 1/min(singular_value)
exercise_19 <- function(A){
  two_norm_matrix <- max(svd(A)$d)
  two_norm_matrix_inverse <- 1/min(svd(A)$d)

  return(two_norm_matrix/two_norm_matrix_inverse)
}
exercise_19_result <- exercise_19(matrix(seq(4,1), 2, 2))
print(paste("The condition number of A is", exercise_19_result))


# Exercise 21
# The value of the square root of the machine precision is 1.490116e-8
exercise_21 <- function(e){
  A <- matrix <- matrix(c(1, 1 - e, 1 + e, 1), 2, 2)
  b <- c( 1 + e + e^2, 1)
  
  
  cmult <- function(A, v) sweep(A, 2, v, `*`)
  
  
  result <- list()
  
  result$manual_solve <- solve(A) %*% b
  result$regular_solve <- matrix(solve(A, b),2,1)
  result$svd_solve <- cmult(svd(A)$v, 1/svd(A)$d) %*% crossprod(svd(A)$u, b)
  result$eigen_solve <- cmult(eigen(A)$vectors, 1/eigen(A)$values) %*% crossprod(eigen(A)$vectors, b)
  
  tryCatch({
  result$qr_solve <- qr.solve(A, b)
  },
  error = function(e){
    print("R is so bad at computation, that in can't do QR decomposition for a small e, it thinks A is singular")
  })
  
  
  result <- lapply(result, function(x) x - matrix(c(1,e),2, 1))
  return(result)
}
exercise_21_result <- exercise_21(1e-6)
print('Errors are:')
print(exercise_21_result)
# solve() produces the best results


# Exercise 26
exercise_26 <- function(vector){
  helper_function <- function(n){
    A <- matrix(rep(-1, n*n), n ,n)
    A[!upper.tri(A)] <- 0
    diag(A) <- 1
    return(max(svd(A)$d)/min(svd(A)$d))
  }
  ratios <- sapply(vector, helper_function)
  return(ratios)
}
exercise_26(3:8)
# It is quite clear that the ratio between the max and min singular value starts growing very fast, nearly exponentially


# Exercise 28
# The columns of U whose corresponding singular values in D are not 0 form an orthonormal basis on A
exercise_28_orthonormal <- function(A){
  singular_vectors <- svd(A)$u
  singular_values <- svd(A)$d
  non_zero_singular_values <- which(singular_values > .Machine$double.eps)

  return(singular_vectors[, non_zero_singular_values])
}
exercise_28_orthonormal(matrix(seq(1,25),5 ,5))


# The columns of V whose corresponding singular values in D are 0 are what we are looking for
exercise_28_kernel <- function(A){
  singular_vectors <- svd(A)$v
  singular_values <- svd(A)$d
  zero_singular_values <- which(singular_values <= .Machine$double.eps)
  
  return(singular_vectors[, zero_singular_values])
}
exercise_28_kernel(matrix(seq(0,96, 2), 7, 7))


# Exercise 31

exercise_31 <- function(n){
  
  vec <- function(A){
    
    output <- c()
    
    for (i in 1:ncol(A)){
      
      output <- c(output, A[, i])
    }
    return(output)
  }
  vech <- function(A){
    output <- c()
    
    for (i in 1:ncol(A)){
      output <- c(output, A[i:nrow(A), i])
    }
    return(output)
  }
  
  # We will use this matrix to create the duplicate matrix, it is an unconventional way of doing things
  A <- matrix(list(), n, n)
  for(i in 1:n){
    for(j in 1:n){
      A[[i,j]] <- c(i,j)
    }
  }
  
  duplicate_matrix <- matrix(0, n^2, (n * (n+1))/2)
  
  vec_posistions <- vec(A)
  vech_positions <- vech(A)


  # The idea is that we go over each element of A and check if it is both present in vec and vech
  # If so its position in vec is its row number in the duplicate matrix and its position in vech is its
  # column position in the duplicate matrix and we are going to assign a value of 1 to this cell,
  # Also when we find an element with i != j we do this operation but we also find its mirror element
  # i.e. the one with j,i index and find its position in the elimination matrix the same way
  for(i in 1:n){
      
    for (j in 1:n){
      current_position <- c(i, j)
     
      # Every element of A is always present in vec
      position_in_vec <- which(sapply(vec_posistions, function(x) identical(x, current_position)) == TRUE)

      # Check if the element of A is present in vech
      present_in_vech <- (i >= j)
      
      position_in_vech <- which(sapply(vech_positions, function(x) identical(x, current_position)) == TRUE)

      # If the element is diagonal Duplicate[i,j] <- 1, i is the index of the element in vec and j index of the 
      # element in vech
      if(i == j){
        duplicate_matrix[position_in_vec, position_in_vech] <- 1
      
      }else{
        
        # If the element is part of vech then Duplicate[vec_index, vech_index] <- 1,
        # as well as we find the symmetrical point, i.e. if the current point had coordinates i,j
        # th symmetric point has coordicnates j,i we find its vec_index and Duplicate[vec_index_symmetric, vech_index] <- 0
        if(present_in_vech == TRUE){
          
          position_in_vech <- which(sapply(vech_positions, function(x) identical(x, current_position)) == TRUE)
          duplicate_matrix[position_in_vec, position_in_vech] <- 1

          symmetric_current_position <- c(j, i)
          symmetric_position_in_vec <- which(sapply(vec_posistions, function(x) identical(x, symmetric_current_position)) == TRUE)
          
          duplicate_matrix[symmetric_position_in_vec, position_in_vech] <- 1

          
        }
          }
      }
      
    }
  return(duplicate_matrix)
  
}
exercise_31(3)


# Exercise 32
exercise_32 <- function(n){
  duplication_matrix <- exercise_31(n)
  
  singular_values <- svd(duplication_matrix)$d
  print(singular_values)
}
exercise_32(10)
# The singular values take only two values 1.414213 and 1.0, regardless of n, for bigger n the only additional
# singular values are 1.414214


#Exercise 33
exercise_33 <- function(n){
  
  vec <- function(A){
    
    output <- c()
    
    for (i in 1:ncol(A)){
      
      output <- c(output, A[, i])
    }
    return(output)
  }

  # We will use this matrix to create the elimination matrix, it is an unconventional way of doing things
  A <- matrix(list(), n, n)
  for(i in 1:n){
    for(j in 1:n){
      A[[i,j]] <- c(i,j)
    }
  }
  
  elimination_matrix <- matrix(0, (n * (n+1))/2, n^2)
  
  vec_posistions <- vec(A)

  # The elimination matrix is much easier to construct, here every row is all 0 with the exception of one cell per row
  # whose column value is equal to the position of teh given element of the lower triangular matrix in vec
  itt <- 0
  for(i in 1:n){
    for(j in 1:n){
      if(i >= j){
        itt <- itt + 1
        current_position <- c(i, j)
        position_in_vec <-  which(sapply(vec_posistions, function(x) identical(x, current_position)) == TRUE)
        elimination_matrix[itt, position_in_vec] <- 1
      }
    }
  }
  return(elimination_matrix)
}
exercise_33(3)
  

# Exercise 34
exercise_34 <- function(n){
  elimination_matrix <- exercise_33(n)
  singular_values <- svd(elimination_matrix)$d
  return(singular_values)
}
exercise_34(5)
# All its singular values are 1s



# Exercise 35
exercise_35 <- function(m,n){
  K <- matrix(0, m*n, m*n)
  
  # The algorithm after googling for finding the positions is vec and t(vec) is
  # that the column number of K is i + m * ( j - 1), while the row number is
  # j + n * (i - 1)
  
  for(i in 1:m){
    
    for(j in 1:n){
      K_column <- i + m * (j- 1)
      K_row <- j + n * (i- 1)
      
      K[K_row, K_column] <- 1
    }
  }

  return(K)
}
exercise_35(2,3)


# Exercise 36
exercise_36 <- function(m,n){
  K <- exercise_35(m,n)
  singular_values <- svd(K)$d
  return(singular_values)
}
exercise_36(3,3)

# The singular values are all 1s


# Exercise 37

# The Moon-Penrose inverse relates to the SVD by the following:
# Moon-Penrose /A+/ = V %*% D+ %*% t(U), where D+ is the diagonal matrix where all the non-zero entries have been inverted
exercise_37 <- function(A, tol){
  
    # This function would be used to compute D+
    helper_function <- function(x, tol){
      if(x <= tol){
        return(0)
      }else{
        return(1/x)
      }
    }
  
    svd_decomposition <- svd(A)
    U <- svd_decomposition$u
    D <- svd_decomposition$d
    V_t <- svd_decomposition$v
    V <- t (V_t)
    
    D_plus <- diag(sapply(D, function(x) helper_function(x, tol)))
    A_plus <- V %*% D_plus %*% t(U)
    return(A_plus)
}
exercise_37(matrix(seq(1,16), 4, 4), 1e-6)



# Exercise 38
exercise_38 <- function(A, w){
  if (ncol(A) == ncol(A)){
    # If A is a square matrix, then the trace of AAt is the square sum of all the elements of A
    return(sum(A^2) * sum(w))
  }
  
  # The trace of A %*% t(A) is equal to the sum of the eigen values of AAt
  trace_A <- sum(eigen(A)$values) * sum(eigen(t(A))$values) * sum(w)
  return(trace_A)
}
exercise_38(matrix(seq(1,81), 9, 9), seq(1,9999))



# Exercise 39
exercise_39 <- function(x, m, V){
  denominator <- (prod(eigen(V)$values) * 2 * pi) ^ -0.5
  
  power <- apply(x, 1, function(xi) {
    diff <- xi - m
    t(diff) %*% solve(V) %*% diff
  })
  
  
  result <-  denominator* exp(-0.5 * power)
  return(result)
}
exercise_39(matrix(c(0, 0, 1, 2, -1, -1), nrow=3, byrow=TRUE), c(0, 0), matrix(c(1, 0.5, 0.5, 1),2, 2))
