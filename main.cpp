#include <iostream>
#include "mkl.h"
#include "mathimf.h"

using namespace std;

void printArray( int array[] , int size);
void initializeMatrix( double * Matrix, int rows, int columns );
void printMatrix( double *Matrix , int rows, int cols );

//void activationFunction( double *a );
void activationFunction( );

int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;

      //--------------------------------------------------------------------
      double EPSILON = 0.01; //learning rate
      double LAMBDA = 0.01;  //regularizer strength

      cout << "Learning rate is " << EPSILON << endl;
      cout << "Regularizer strength is " << LAMBDA << endl;
      //--------------------------------------------------------------------
      int targets[10] = {0};
      //set values
      targets[0] = 0; targets[5] = 1;
      targets[1] = 1; targets[6] = 0;
      targets[2] = 1; targets[7] = 1;
      targets[3] = 0; targets[8] = 0;
      targets[4] = 1; targets[9] = 1;

      cout << "targets are :" << endl;
      printArray( targets , 10 );
      //--------------------------------------------------------------------
      int NUM_SAMPLES = 10;
      int NUM_FEATURES = 2;
      double * features = (double *)mkl_malloc( NUM_SAMPLES* NUM_FEATURES*sizeof( double ), 64 );
      initializeMatrix( features, NUM_SAMPLES, NUM_FEATURES );

      //set features for now
      features[0]	=  0.74346118;  features[1]  =  0.46465633;
      features[2]	=  1.65755662;  features[3]  = -0.63203157;
      features[4]	= -0.15878875;  features[5]  =  0.25584465;
      features[6]	= -1.088752  ;  features[7]  = -0.39694315;
      features[8]	=  1.768052  ;  features[9]  = -0.25443213;
      features[10]	=  1.95416454;  features[11] = -0.12850579;
      features[12]	=  0.93694537;  features[13] =  0.36597075;
      features[14]	=  0.88446589;  features[15] = -0.47595401;
      features[16]	=  0.80950246;  features[17] =  0.3505231 ;
      features[18]	=  1.2278091;   features[19] = -0.64785108;

      printMatrix( features, NUM_SAMPLES, NUM_FEATURES );
      //--------------------------------------------------------------------
      //test activation function!
      //activationFunction( 5.0 );
      activationFunction( );
      //--------------------------------------------------------------------
      //--------------------------------------------------------------------
      //--------------------------------------------------------------------
      return 0;
}

void printArray( int array[] , int size){

      for (int iter = 0; iter < size - 1; ++iter) 
	    {
		  cout << array[iter] << " , ";
	    }
      cout << array[(size - 1)] << endl;
}

void initializeMatrix( double * Matrix , int rows, int cols ){
      printf (" Intializing matrix data\n");
      for (int i = 0; i < (rows*cols); i++) {
	    Matrix[i] = (double)(0.0);
      }
}

void printMatrix( double *Matrix , int rows, int cols ){
      //      printf ("Features matrix: \n");
      for ( int i = 0; i < rows ; i++) {
	    for ( int j = 0; j < cols ; j++) {
		  printf ("%12.5f", Matrix[i*cols + j]);
	    }
	    printf ("\n");
      }
}

//void activationFunction( double *a ){
void activationFunction( ){
      /*
      //logistic sigmoid
      double value = 0.0;

      value = 1/( 1 + exp( -a ) );

      cout <<"Value " << a << " is transformed to " << value << endl;
      */
      double * a = (double *)mkl_malloc( 1 * 2*sizeof( double ), 64 );
      double * y = (double *)mkl_malloc( 1 * 2*sizeof( double ), 64 );
      double * v = (double *)mkl_malloc( 1 * 2*sizeof( double ), 64 );
      for (int i = 0; i < (1*2); i++) {
	    a[i] = (double)(0.0);
	    y[i] = (double)(0.0);
	    v[i] = (double)(0.0);
      }
      a[0] = 1.0;
      a[1] = 2.0;

      cout << "A is " << endl;
      for (int i = 0; i < (1*2); i++) {
	    cout << a[i] <<" , " ;
      }
      cout << endl;

      //multiply by -1
      for (int i = 0; i < (1*2); i++) {
	    a[i] = a[i] * (-1);
      }

      //calculate exponential!
      vdExp( 2, a, y );

      cout << "y is " << endl;
      for (int i = 0; i < (1*2); i++) {
	    cout << y[i] <<" , " ;
      }
      cout << endl;

      //compute activation!
      for (int i = 0; i < (1*2); i++) {
	    v[i] = 1/( 1 + y[i]);
      }

      cout << "v is " << endl;
      for (int i = 0; i < (1*2); i++) {
	    cout << v[i] <<" , " ;
      }
      cout << endl;
}

/* extended to opertating on a matrix of data
void activationFunction( ){
      double * a = (double *)mkl_malloc( 2 * 2*sizeof( double ), 64 );
      double * y = (double *)mkl_malloc( 2 * 2*sizeof( double ), 64 );
      for (int i = 0; i < (2*2); i++) {
	    a[i] = (double)(0.0);
	    y[i] = (double)(0.0);
      }
      a[0]	=  0.74346118;  a[1]  =  0.46465633;
      a[2]	=  1.65755662;  a[3]  = -0.63203157;

      cout << "A is " << endl;
      printMatrix( a , 2, 2);

      //calculate exponential!
      //multiply by -1
      for (int i = 0; i < (2*2); i++) {
	    a[i] = a[i] * (-1);
      }

      vdExp( 4, a, y );
      cout << "Y is " << endl;
      printMatrix( y , 2 , 2);
}
*/
