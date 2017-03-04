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
void printMatrix( float *Matrix , int rows, int cols );
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
      //      cout << "entropy of system is " << entropy << endl;
      return entropy;
}

void errorFunction(float *t , float *y);
//void crossEntropyFunction( float *t, float *y );
//-----------------------------------------------------
//-----------------------------------------------------
float *activationFunction( float *a );

float fRand(float fMin, float fMax);
float *getHiddenActivations( float * x, float * firstLayerWeightMatrix );
float *getOutputActivations( float * x, float * secondLayerWeightVector );

void logisticSigmoid( float * finalOutputs );
void dlogisticSigmoid( float * a );

int main(int argc, char *argv[])
{
      cout << " Creating a Simple Neural Net" << endl;

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
      x[1] =  0.46465633;

      t[0] = 0;

      cout << "Features are " << endl;
      printMatrix( x, 1, NUM_FEATURES );

      cout << "t are :" << endl;
      printMatrix( t, 1, NUM_SAMPLES);
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

      //--------------------------------------------------------------------
      // test MLP functions
      // cross entropy function
      y[0] = 0.5;
      cout << "Entropy of system is " << crossEntropyFunction( t , y ) << "\n";

      exit( -1 );
      //--------------------------------------------------------------------
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
      logisticSigmoid( finalOutputs );
      //--------------------------------------------------------------------
      //get derivative of sigmoid!
      dlogisticSigmoid( finalOutputs );
      return 0;
}
void logisticSigmoid( float * finalOutputs ){
      cout << "Transforming with sigmoid func" << endl;

      //compute activation!
      for (int i = 0; i < (NUM_OUTPUTS); i++) {
	    finalOutputs[i] = 1/( 1 + exp(-finalOutputs[i]));
      }
      cout << "value is " << finalOutputs[0] << endl;
}

void dlogisticSigmoid( float * a ){
      //d/da sigma(a) = sigma(a) * [1 - sigma(a)]
      cout << "calculating derivative of Logistic Sigmoid" << "\n";

      double sigma = 0.0;

      //sigma = logisticSigmoid( a );

      sigma = sigma * (1 - sigma);

      cout << "sigma is" << sigma << "\n";
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
