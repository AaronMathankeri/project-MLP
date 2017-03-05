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
myMatrix = matrix( c(7.99692,     4.26953,
                     -1.99978,     9.63194,
                     3.95212,     3.22824) , nrow = 3, ncol =2, byrow = TRUE )
myVector <-c(0.74346   ,  0.46466)
myMatrix %*% myVector

#------
sigmoid <- function( a ){
  y = 1/(1 + exp(-a) );
}
#------
crossEntropy <- function( t , y){
  ce = 0.0;
  ce= t*log(y) + (1 -t)*log(1 -y)
 
  return(ce) 
}

t = 0
y = 
ce <- crossEntropy( t , y)

ce