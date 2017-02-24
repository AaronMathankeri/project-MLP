#include <iostream>
#include "mkl.h"
#include "mathimf.h"

using namespace std;
int NUM_SAMPLES = 10;
int NUM_FEATURES = 2;

void printArray( int array[] , int size);
void printMatrix( double *Matrix , int rows, int cols );

void initializeMatrix( double * Matrix, int rows, int columns );

void activationFunction( double *a );
void errorFunction( );

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
      activationFunction( features );
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

void activationFunction( double *a ){

      double * Z = (double *)mkl_malloc( NUM_SAMPLES * NUM_FEATURES*sizeof( double ), 64 );
      initializeMatrix( Z , NUM_SAMPLES , NUM_FEATURES);

      //calculate exponential!
      //multiply by -1
      for (int i = 0; i < (NUM_SAMPLES*NUM_FEATURES); i++) {
	    a[i] = a[i] * (-1);
      }

      vdExp( (NUM_SAMPLES*NUM_FEATURES), a, a );

      //compute activation!
      for (int i = 0; i < (NUM_SAMPLES*NUM_FEATURES); i++) {
	    Z[i] = 1/( 1 + a[i]);
      }

      cout << "Z is " << endl;
      printMatrix( Z , NUM_SAMPLES , NUM_FEATURES);
}

