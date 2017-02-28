#include <iostream>
#include "mkl.h"
#include "mathimf.h"
#include <random>
#include <cstdlib>

using namespace std;
const int NUM_SAMPLES = 1;
const int NUM_FEATURES = 2;
const int NUM_HIDDEN_NODES = 3;
const int NUM_OUTPUTS = 1;

void printMatrix( double *Matrix , int rows, int cols );
void initializeMatrix( double * Matrix, int rows, int columns );
void initializeWeightMatrix( double * weightMatrix );
void initializeWeightVector( double * weightVector );

double *activationFunction( double *a );
void errorFunction(double *t , double *y);
double fRand(double fMin, double fMax);
double *getHiddenActivations( double * features, double * firstLayerWeightMatrix );
double *getOutputActivations( double * features, double * outputLayerWeightVector );

int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;

      //--------------------------------------------------------------------
      double EPSILON = 0.01; //learning rate
      double LAMBDA = 0.01;  //regularizer strength
      //cout << "Learning rate is " << EPSILON << endl;
      //cout << "Regularizer strength is " << LAMBDA << endl;
      //--------------------------------------------------------------------
      double * features = (double *)mkl_malloc( NUM_FEATURES*sizeof( double ), 64 );
      double * firstLayerWeightMatrix = (double *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( double ), 64 );
      double * outputLayerWeightVector = (double *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( double ), 64 );
      double * targets = (double *)mkl_malloc( NUM_SAMPLES*sizeof( double ), 64 );
      
      //hard-code some values
      features[0] = 0.74346118;
      features[1] =  0.46465633;
      targets[0] = 0;

      cout << "Features are " << endl;
      printMatrix( features, 1, NUM_FEATURES );

      cout << "targets are :" << endl;
      printMatrix( targets, 1, NUM_SAMPLES);
      //--------------------------------------------------------------------
      // randomly initialize weight matrices
      srand(time(NULL)); //set seed
      initializeMatrix( firstLayerWeightMatrix,  NUM_HIDDEN_NODES, NUM_FEATURES );
      initializeMatrix( outputLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      initializeWeightMatrix( firstLayerWeightMatrix );
      initializeWeightVector( outputLayerWeightVector );

      cout << "1st layer Weight Matrix" << endl;
      printMatrix( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );
      cout << "Output layer Weight Matrix" << endl;
      printMatrix( outputLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------

      //--------------------------------------------------------------------
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      double *activations = getHiddenActivations( features , firstLayerWeightMatrix );
      cout << "Hidden activations are"<< endl;
      printMatrix( activations , 1 , NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------

      //2. transform with logistic sigmoid
      double *z = activationFunction( activations );
      cout << "Non-linear transformation complete. Z is " << endl;
      printMatrix( z , 1 , NUM_HIDDEN_NODES);
      //--------------------------------------------------------------------

      //compute final layer
      double *finalOutputs = getOutputActivations( z , outputLayerWeightVector );
      cout << "finalOutputs are :" << endl;
      printMatrix( finalOutputs, 1, NUM_OUTPUTS);
      //--------------------------------------------------------------------
      //test error function
      errorFunction( targets , finalOutputs );
      //--------------------------------------------------------------------
      return 0;
}

double *getOutputActivations( double * z, double * outputLayerWeightVector ){
      cout << "Computing final output" << endl;
      double * finalOutputs = (double *)mkl_malloc( NUM_OUTPUTS*sizeof( double ), 64 );

      initializeMatrix( finalOutputs, 1 , NUM_OUTPUTS );
      // perform matrix vector multiplication: y = W*z
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      //cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES, alpha, outputLayerWeightVector, NUM_HIDDEN_NODES, z, incx, beta, finalOutputs, incx);
      double res = 0.0;
      res = cblas_ddot( NUM_HIDDEN_NODES, z,incx, outputLayerWeightVector, incx);

      finalOutputs[0] = res;
      return finalOutputs;
}
double *getHiddenActivations( double * features, double * weightMatrix ){
      cout << "Computing 1st layer" << endl;
      double * activations = (double *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( double ), 64 );
      initializeMatrix( activations, 1 , NUM_HIDDEN_NODES );

      // perform matrix vector multiplication: a = W*x
      const double alpha = 1.0;
      const double beta = 0.0;
      const int incx = 1;
      cblas_dgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES, alpha, weightMatrix, NUM_FEATURES, features, incx, beta, activations, incx);

      return activations;
}

void initializeMatrix( double * Matrix , int rows, int cols ){
      for (int i = 0; i < (rows*cols); i++) {
	    Matrix[i] = (double)(0.0);
      }
}

void initializeWeightMatrix( double * weightMatrix ){
      cout << "Random Initialization of Weight Matrix" << endl;
      for (int iter = 0; iter < (NUM_HIDDEN_NODES * NUM_FEATURES); ++iter) {
	    double temp = fRand( -10.0, 10.0);
	    weightMatrix[iter] = temp;
      }
}
void initializeWeightVector( double * weightVector ){
      cout << "Random Initialization of Weight Matrix" << endl;
      for (int iter = 0; iter < (NUM_HIDDEN_NODES); ++iter) {
	    double temp = fRand( -10.0, 10.0);
	    weightVector[iter] = temp;
      }
}

double fRand(double fMin, double fMax){
      double f = (double)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}

void printMatrix( double *Matrix , int rows, int cols ){
      for ( int i = 0; i < rows ; i++) {
	    for ( int j = 0; j < cols ; j++) {
		  printf ("%12.5f", Matrix[i*cols + j]);
	    }
	    printf ("\n");
      }
}

double * activationFunction( double *a ){
      //sigma(a) = 1/(1 + exp(-a))

      double * z = (double *)mkl_malloc( NUM_HIDDEN_NODES * sizeof( double ), 64 );
      initializeMatrix( z , 1 , NUM_HIDDEN_NODES);

      //multiply by -1
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    a[i] = a[i] * (-1);
      }

      vdExp( (NUM_HIDDEN_NODES), a, a );

      //compute activation!
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    z[i] = 1/( 1 + a[i]);
      }
      return z;
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
//double * features = (double *)mkl_malloc( NUM_SAMPLES* NUM_FEATURES*sizeof( double ), 64 );
//initializeMatrix( features, NUM_SAMPLES, NUM_FEATURES );

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

