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