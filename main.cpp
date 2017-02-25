#include <iostream>
#include "mkl.h"
#include "mathimf.h"

using namespace std;
const int NUM_SAMPLES = 1;
const int NUM_FEATURES = 2;

void printArray( int array[] , int size);
void printMatrix( double *Matrix , int rows, int cols );

void initializeMatrix( double * Matrix, int rows, int columns );

void activationFunction( double *a );
void errorFunction(double *t , double *y);

int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;

      //--------------------------------------------------------------------
      double EPSILON = 0.01; //learning rate
      double LAMBDA = 0.01;  //regularizer strength

      cout << "Learning rate is " << EPSILON << endl;
      cout << "Regularizer strength is " << LAMBDA << endl;
      //--------------------------------------------------------------------
      int targets[1] = {0};

      //set values
      cout << "targets are :" << endl;
      printArray( targets , 1 );
      //--------------------------------------------------------------------
      //double * features = (double *)mkl_malloc( NUM_SAMPLES* NUM_FEATURES*sizeof( double ), 64 );
      double * features = (double *)mkl_malloc( NUM_FEATURES*sizeof( double ), 64 );
      //initializeMatrix( features, NUM_SAMPLES, NUM_FEATURES );

      //set features for now
      features[0]	=  0.74346118;  features[1]  =  0.46465633;
      cout << "Features are " << endl;
      printMatrix( features, 1, NUM_FEATURES );

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
      for (int i = 0; i < (rows*cols); i++) {
	    Matrix[i] = (double)(0.0);
      }
}

void printMatrix( double *Matrix , int rows, int cols ){
      for ( int i = 0; i < rows ; i++) {
	    for ( int j = 0; j < cols ; j++) {
		  printf ("%12.5f", Matrix[i*cols + j]);
	    }
	    printf ("\n");
      }
}

void activationFunction( double *a ){
      //sigma(a) = 1/(1 + exp(-a))

      double * z = (double *)mkl_malloc( NUM_FEATURES * sizeof( double ), 64 );
      initializeMatrix( z , 1 , NUM_FEATURES);

      //multiply by -1
      for (int i = 0; i < (NUM_FEATURES); i++) {
	    a[i] = a[i] * (-1);
      }

      vdExp( (NUM_FEATURES), a, a );

      //compute activation!
      for (int i = 0; i < (NUM_FEATURES); i++) {
	    z[i] = 1/( 1 + a[i]);
      }

      cout << "z is " << endl;
      printMatrix( z , 1 , NUM_FEATURES);
}

void errorFunction( double *targets, double *finalOutputs){

      cout << "Error function for training" << endl;
      double * diff = (double *)mkl_malloc( NUM_SAMPLES * sizeof( double ), 64 );
      double res = 0.0;
      const int incx = 1;
      
      for (int i = 0; i < (NUM_SAMPLES); i++) {
	    diff[i] = (double)(0.0);
      }

      //subtract them!
      vdSub( NUM_SAMPLES , targets , finalOutputs, diff);

      //get norm
      res = dnrm2( &(NUM_SAMPLES), diff, &incx);

      //get norm squared
      res *= res;

      cout << "Error = " << res << endl;
}
      /*
      features[2]	=  1.65755662;  features[3]  = -0.63203157;
      features[4]	= -0.15878875;  features[5]  =  0.25584465;
      features[6]	= -1.088752  ;  features[7]  = -0.39694315;
      features[8]	=  1.768052  ;  features[9]  = -0.25443213;
      features[10]	=  1.95416454;  features[11] = -0.12850579;
      features[12]	=  0.93694537;  features[13] =  0.36597075;
      features[14]	=  0.88446589;  features[15] = -0.47595401;
      features[16]	=  0.80950246;  features[17] =  0.3505231 ;
      features[18]	=  1.2278091;   features[19] = -0.64785108;
      */
      /*
      targets[1] = 1; targets[6] = 0;
      targets[2] = 1; targets[7] = 1;
      targets[3] = 0; targets[8] = 0;
      targets[4] = 1; targets[9] = 1;
      */

