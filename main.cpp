#include <iostream>
#include "mkl.h"
#include "mathimf.h"

using namespace std;

const int NUM_SAMPLES = 1;
const int NUM_FEATURES = 2;
const int NUM_HIDDEN_NODES = 3;
const int NUM_OUTPUTS = 1;

//-----------------------------------------------------
//I-O functions
void printMatrix( float *Matrix , int nRows, int nCols ){
      for ( int i = 0; i < nRows ; i++) {
	    for ( int j = 0; j < nCols ; j++) {
		  printf ("%12.5f", Matrix[i*nCols + j]);
	    }
	    printf ("\n");
      }
}
//-----------------------------------------------------
//initialize to avoid memory errors
void initializeMatrix( float * Matrix, int rows, int columns ){
      memset( Matrix, 0.0,  rows*columns * sizeof(double));      
}
float fRand(float fMin, float fMax){
      float f = (float)rand() / RAND_MAX;
      return fMin + f * (fMax - fMin);
}
void setRandomWeights( float * weights, int nRows, int nCols ){
      for (int i = 0; i < (nRows*nCols); ++i) {
	    float temp = fRand( -10.0, 10.0);
	    weights[i] = temp;
      }
}
//-----------------------------------------------------
// MLP network Independent functions
float crossEntropyFunction( float *t, float *y ){
      float entropy = 0.0;
      for (int i = 0; i < NUM_SAMPLES; ++i){
	    entropy += -(t[i]*log(y[i]) + (1 - t[i])*log(1 - y[i]));		  
      }
      return entropy;
}
void logisticSigmoid( float * a , float *sigma, int length){
      //sigma(a) = 1/(1 + exp(-a))
      for (int i = 0; i < (length); i++) {
	    sigma[i] = 1/( 1 + exp(-a[i]));
      }
}
void dlogisticSigmoid( float * a , float * sigmaPrime, int length){
      //d/da sigma(a) = sigma(a) * [1 - sigma(a)]
      logisticSigmoid(a, sigmaPrime, length);
      for (int i = 0; i < (length); ++i) {
	    sigmaPrime[i] = sigmaPrime[i]*(1 - sigmaPrime[i] );
      }
}
//-----------------------------------------------------
// MLP network Dependent functions
void computeActivations( float* x, float* firstLayerWeightMatrix, float* a){
      cout << "Computing 1st Layer Activations" << "\n";
      // perform matrix vector multiplication: a = W*x
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      cblas_sgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES,
		   alpha, firstLayerWeightMatrix, NUM_FEATURES, x, incx, beta, a, incx);
}
void computeHiddenUnits( float* a, float* z, int length){
      logisticSigmoid( a , z , length);
}

float *activationFunction( float *a );
float *getHiddenActivations( float * x, float * firstLayerWeightMatrix );
float *getOutputActivations( float * x, float * secondLayerWeightVector );
//-----------------------------------------------------
int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      float * x = (float *)mkl_malloc( NUM_FEATURES*sizeof( float ), 64 );
      float * firstLayerWeightMatrix = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * secondLayerWeightVector = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float * t = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 );
      float * y = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 );

      initializeMatrix( x, NUM_FEATURES , 1);
      initializeMatrix( t, NUM_SAMPLES , 1);
      initializeMatrix( y, NUM_SAMPLES , 1);

      //hard-code some values
      x[0] = 0.74346118;
      x[1] = 0.46465633;

      t[0] = 0.0;

      cout << "Features are " << endl;
      printMatrix( x, 1, NUM_FEATURES );

      cout << "t are :" << endl;
      printMatrix( t, 1, NUM_SAMPLES);
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      // randomly initialize weight matrices
      srand(time(NULL)); //set seed
      initializeMatrix( firstLayerWeightMatrix,  NUM_HIDDEN_NODES, NUM_FEATURES );
      initializeMatrix( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      setRandomWeights( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );
      setRandomWeights( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      cout << "1st layer Weight Matrix" << endl;
      printMatrix( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );
      cout << "Output layer Weight Matrix" << endl;
      printMatrix( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      // test MLP functions
      // cross entropy function
      y[0] = 0.5;
      cout << "Entropy of system is " << crossEntropyFunction( t , y ) << "\n";

      float * test = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 );
      initializeMatrix( test, 1 , NUM_SAMPLES);
	    
      logisticSigmoid( t, test, NUM_OUTPUTS );
      cout <<"Input of sigmoid " << t[0] << "\n";
      cout <<"Output of Logistic Sigmoid is " << test[0] << "\n";
      initializeMatrix( test, 1 , NUM_SAMPLES);
      dlogisticSigmoid( t, test, NUM_OUTPUTS );
      cout <<"Input of dSigmoid " << t[0] << "\n";
      cout <<"Output of dLogistic Sigmoid is " << test[0] << "\n";
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      // test MLP topology transformations
      float * a = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      float * z = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      initializeMatrix( a, NUM_HIDDEN_NODES , 1);
      initializeMatrix( z, NUM_HIDDEN_NODES , 1);

      computeActivations( x, firstLayerWeightMatrix, a);
      computeHiddenUnits( a , z, NUM_HIDDEN_NODES );
      
      cout << "Activations are:" << "\n";
      printMatrix( a, NUM_HIDDEN_NODES, 1);
      cout << "Hidden Units are:" << "\n";
      printMatrix( z, NUM_HIDDEN_NODES, 1);

      exit( -1 );
      /*
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      float *activations = getHiddenActivations( x , firstLayerWeightMatrix );
      cout << "Hidden activations are"<< endl;
      printMatrix( activations , 1 , NUM_HIDDEN_NODES );
      //--------------------------------------------------------------------

      //2. transform with logistic sigmoid
      float *z = activationFunction( activations );
      cout << "Non-linear transformation complete. Z is " << endl;
      printMatrix( z , 1 , NUM_HIDDEN_NODES);
      //--------------------------------------------------------------------

      //compute final layer
      float *finalOutputs = getOutputActivations( z , secondLayerWeightVector );
      cout << "finalOutputs are :" << endl;
      printMatrix( finalOutputs, 1, NUM_OUTPUTS);
      //--------------------------------------------------------------------
      //test error function
      //errorFunction( t , finalOutputs );
      // take output and run it through sigmoid func!
      //logisticSigmoid( finalOutputs );
      //--------------------------------------------------------------------
      //get derivative of sigmoid!
      //dlogisticSigmoid( finalOutputs );
      */
      return 0;
}



float *getOutputActivations( float * z, float * secondLayerWeightVector ){
      cout << "Computing final output" << endl;
      float * finalOutputs = (float *)mkl_malloc( NUM_OUTPUTS*sizeof( float ), 64 );

      initializeMatrix( finalOutputs, 1 , NUM_OUTPUTS );
      // perform matrix vector multiplication: y = W*z
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      //cblas_dgemv( CblasRowMajor, CblasTrans, NUM_OUTPUTS, NUM_HIDDEN_NODES, alpha, secondLayerWeightVector, NUM_HIDDEN_NODES, z, incx, beta, finalOutputs, incx);
      float res = 0.0;
      res = cblas_sdot( NUM_HIDDEN_NODES, z,incx, secondLayerWeightVector, incx);

      finalOutputs[0] = res;
      return finalOutputs;
}
float *getHiddenActivations( float * x, float * weightMatrix ){
      cout << "Computing 1st layer" << endl;
      float * activations = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      initializeMatrix( activations, 1 , NUM_HIDDEN_NODES );

      // perform matrix vector multiplication: a = W*x
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      cblas_sgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES,
		   alpha, weightMatrix, NUM_FEATURES, x, incx, beta, activations, incx);

      return activations;
}


