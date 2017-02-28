#x <- c(1 , 2)
x <- c( 0.74346118 , 0.46465633)
#x = matrix( c(0.74346118 , 0.46465633, 1.65756  ,  -0.63203) , nrow = 2, ncol =2, byrow = TRUE )

myFunc <- function( x ){
  
#  exp(-x)
  y = 1/(1 + exp(-x))
 
   return(y) 
}

a<- myFunc(x)

a

#------------
# matrix vector multiplication
myMatrix = matrix( c(2.27326 , 6.75528, -3.92609  ,  -5.75392,-6.13284   ,  5.29003) , nrow = 3, ncol =2, byrow = TRUE )
myVector <-c(0.74346   ,  0.46466)
myMatrix %*% myVector