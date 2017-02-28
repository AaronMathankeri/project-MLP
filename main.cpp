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

void printMatrix( float *Matrix , int rows, int cols );
void initializeMatrix( float * Matrix, int rows, int columns );
void initializeWeightMatrix( float * weightMatrix );
void initializeWeightVector( float * weightVector );

float *activationFunction( float *a );
void errorFunction(float *t , float *y);
float fRand(float fMin, float fMax);
float *getHiddenActivations( float * features, float * firstLayerWeightMatrix );
float *getOutputActivations( float * features, float * outputLayerWeightVector );

int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;

      //--------------------------------------------------------------------
      float EPSILON = 0.01; //learning rate
      float LAMBDA = 0.01;  //regularizer strength
      //cout << "Learning rate is " << EPSILON << endl;
      //cout << "Regularizer strength is " << LAMBDA << endl;
      //--------------------------------------------------------------------
      float * features = (float *)mkl_malloc( NUM_FEATURES*sizeof( float ), 64 );
      float * firstLayerWeightMatrix = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * outputLayerWeightVector = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float * targets = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 );
      
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
      float *activations = getHiddenActivations( features , firstLayerWeightMatrix );
      cout << "Hidden activations are"<< endl;
      printMatrix( activations , 1 , NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------

      //2. transform with logistic sigmoid
      float *z = activationFunction( activations );
      cout << "Non-linear transformation complete. Z is " << endl;
      printMatrix( z , 1 , NUM_HIDDEN_NODES);
      //--------------------------------------------------------------------

      //compute final layer
      float *finalOutputs = getOutputActivations( z , outputLayerWeightVector );
      cout << "finalOutputs are :" << endl;
      printMatrix( finalOutputs, 1, NUM_OUTPUTS);
      //--------------------------------------------------------------------
      //test error function
      errorFunction( targets , finalOutputs );
      //--------------------------------------------------------------------
      return 0;
}

float *getOutputActivations( float * z, float * outputLayerWeightVector ){
      cout << "Computing final output" << endl;
      float * finalOutputs = (float *)mkl_malloc( NUM_OUTPUTS*sizeof( float ), 64 );

      initializeMatrix( finalOutputs, 1 , NUM_OUTPUTS );
      // perform matrix vector multiplication: y = W*z
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      //cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES, alpha, outputLayerWeightVector, NUM_HIDDEN_NODES, z, incx, beta, finalOutputs, incx);
      float res = 0.0;
      res = cblas_sdot( NUM_HIDDEN_NODES, z,incx, outputLayerWeightVector, incx);

      finalOutputs[0] = res;
      return finalOutputs;
}
float *getHiddenActivations( float * features, float * weightMatrix ){
      cout << "Computing 1st layer" << endl;
      float * activations = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      initializeMatrix( activations, 1 , NUM_HIDDEN_NODES );

      // perform matrix vector multiplication: a = W*x
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      cblas_sgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES,
		   alpha, weightMatrix, NUM_FEATURES, features, incx, beta, activations, incx);

      return activations;
}

void initializeMatrix( float * Matrix , int rows, int cols ){
      for (int i = 0; i < (rows*cols); i++) {
	    Matrix[i] = (float)(0.0);
      }
}

void initializeWeightMatrix( float * weightMatrix ){
      cout << "Random Initialization of Weight Matrix" << endl;
      for (int iter = 0; iter < (NUM_HIDDEN_NODES * NUM_FEATURES); ++iter) {
	    float temp = fRand( -10.0, 10.0);
	    weightMatrix[iter] = temp;
      }
}
void initializeWeightVector( float * weightVector ){
      cout << "Random Initialization of Weight Matrix" << endl;
      for (int iter = 0; iter < (NUM_HIDDEN_NODES); ++iter) {
	    float temp = fRand( -10.0, 10.0);
	    weightVector[iter] = temp;
      }
}

float fRand(float fMin, float fMax){
      float f = (float)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}

void printMatrix( float *Matrix , int rows, int cols ){
      for ( int i = 0; i < rows ; i++) {
	    for ( int j = 0; j < cols ; j++) {
		  printf ("%12.5f", Matrix[i*cols + j]);
	    }
	    printf ("\n");
      }
}

float * activationFunction( float *a ){
      //sigma(a) = 1/(1 + exp(-a))

      float * z = (float *)mkl_malloc( NUM_HIDDEN_NODES * sizeof( float ), 64 );
      initializeMatrix( z , 1 , NUM_HIDDEN_NODES);

      //multiply by -1
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    a[i] = a[i] * (-1);
      }

      vsExp( (NUM_HIDDEN_NODES), a, a );

      //compute activation!
      for (int i = 0; i < (NUM_HIDDEN_NODES); i++) {
	    z[i] = 1/( 1 + a[i]);
      }
      return z;
}

void errorFunction( float *targets, float *finalOutputs){

      cout << "Error function for training" << endl;
      float * diff = (float *)mkl_malloc( NUM_SAMPLES * sizeof( float ), 64 );
      float res = 0.0;
      const int incx = 1;
      
      for (int i = 0; i < (NUM_SAMPLES); i++) {
	    diff[i] = (float)(0.0);
      }

      //subtract them!
      vsSub( NUM_SAMPLES , targets , finalOutputs, diff);

      //get norm
      res = snrm2( &(NUM_SAMPLES), diff, &incx);

      //get norm squared
      res *= res;
      //multiply by -1.0
      res = res * -1.0;
      cout << "Error = " << res << endl;
}
//float * features = (float *)mkl_malloc( NUM_SAMPLES* NUM_FEATURES*sizeof( float ), 64 );
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

