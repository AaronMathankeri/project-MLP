#include <iostream>
#include "mkl.h"
#include "mathimf.h"

using namespace std;

const int NUM_SAMPLES = 1;
const int NUM_FEATURES = 2;
const int NUM_HIDDEN_NODES = 3;
const int NUM_OUTPUTS = 1;
const float LEARNING_RATE = 0.01;
const float THRESH = 0.01;
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
// Feed-forward Functions
void computeActivations( float* x, float* firstLayerWeightMatrix, float* a){
      //1. get activations: a_j = \sum_i^D{w_ji * x_i + w_j0}
      // perform matrix vector multiplication: a = W*x
      cout << "Computing 1st Layer Activations" << "\n";
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      cblas_sgemv( CblasRowMajor, CblasNoTrans, NUM_HIDDEN_NODES, NUM_FEATURES,
		   alpha, firstLayerWeightMatrix, NUM_FEATURES, x, incx, beta, a, incx);
}
void computeHiddenUnits( float* a, float* z, int length){
      logisticSigmoid( a , z , length);
}

void computeOutputActivations( float* z, float* secondLayerWeightVector, float* v){
      // perform matrix vector multiplication: y = W*z
      cout << "Computing 2nd Layer Activations" << "\n";
      const float alpha = 1.0;
      const float beta = 0.0;
      const int incx = 1;
      float res = 0.0;
      res = cblas_sdot( NUM_HIDDEN_NODES, z,incx, secondLayerWeightVector, incx);
      v[(NUM_OUTPUTS - 1)] = res;
}
//-----------------------------------------------------
// BackProp algorithms
float computeOutputErrors( float* y, float* t){
      float delta = 0.0;
      delta = y[(NUM_OUTPUTS - 1)] - t[(NUM_OUTPUTS - 1)];
      return delta;
}
void computeHiddenErrors( float* a, float* secondLayerWeightVector, float delta, float* hiddenDeltas){
      cout << "Computing Hidden Errors" << endl;
      float * dA = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );
      initializeMatrix( dA, NUM_HIDDEN_NODES, 1);

      dlogisticSigmoid( a, dA, NUM_HIDDEN_NODES );
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    secondLayerWeightVector[i] = secondLayerWeightVector[i]*delta;
	    hiddenDeltas[i] = dA[i]*secondLayerWeightVector[i];
      }
}
void computeSecondLayerDerivatives( float* z, float delta, float* secondLayerDerivatives){
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    secondLayerDerivatives[i] = z[i]*delta;
      }
}
void computeFirstLayerDerivatives( float* x, float* hiddenDeltas, float* firstLayerDerivatives){
      for (int i = 0; i < NUM_HIDDEN_NODES; ++i) {
	    for (int j = 0; j < NUM_FEATURES; ++j) {
		  firstLayerDerivatives[i*NUM_FEATURES + j] = hiddenDeltas[i]*x[j];
	    }
      }
}
//-----------------------------------------------------
// Gradient descent updates
void updateFirstLayerWeights( float* firstLayerWeightMatrix, const float* firstLayerDerivatives, const float LEARNING_RATE){
      for (int i = 0; i < (NUM_HIDDEN_NODES*NUM_FEATURES); ++i) {
	    firstLayerWeightMatrix[i] = firstLayerWeightMatrix[i] - (LEARNING_RATE*firstLayerDerivatives[i]);
      }
      cout << "New values for 1st layer Weight Matrix" << endl;
      printMatrix( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );
}
void updateSecondLayerWeights( float* secondLayerWeightVector, const float* secondLayerDerivatives, const float LEARNING_RATE){
      for (int i = 0; i < (NUM_HIDDEN_NODES*NUM_OUTPUTS); ++i) {
	    secondLayerWeightVector[i] = secondLayerWeightVector[i] - (LEARNING_RATE*secondLayerWeightVector[i]);
      }
      cout << "New values for output layer Weight Matrix" << endl;
      printMatrix( secondLayerWeightVector,  NUM_HIDDEN_NODES, NUM_OUTPUTS );
}
//-----------------------------------------------------
int main(int argc, char *argv[])
{
      cout << " CREATING A SIMPLE NEURAL NET" << endl;
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      float * x = (float *)mkl_malloc( NUM_FEATURES*sizeof( float ), 64 ); //features
      float * firstLayerWeightMatrix = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * secondLayerWeightVector = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float * t = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 ); //targets
      float * y = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 ); //predictions
      float * a = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );//activations
      float * z = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );//hidden units
      float * v = (float *)mkl_malloc( NUM_OUTPUTS*sizeof( float ), 64 );//output activations
      float * hiddenDeltas = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 ); //errors from backprop
      float * firstLayerDerivatives = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * secondLayerDerivatives = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float delta = 0.0; //error from initialbackprop

      initializeMatrix( x, NUM_FEATURES , 1);
      initializeMatrix( t, NUM_SAMPLES , 1);
      initializeMatrix( y, NUM_SAMPLES , 1);

      //hard-code some values
      x[0] = 0.22455898;
      x[1] = 0.56056104;
      //x[0] = 1.01417185;
      //x[1] = 0.01460828;
      t[0] = 1.0;
      //t[0] = 0.0;

      cout << "Features are :" << endl;
      printMatrix( x, NUM_FEATURES, 1 );

      cout << "Targets are :" << endl;
      printMatrix( t, NUM_SAMPLES, 1 );
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------


      // randomly initialize weight matrices
      srand(time(NULL)); //set seed
      initializeMatrix( firstLayerWeightMatrix,  NUM_HIDDEN_NODES, NUM_FEATURES );
      initializeMatrix( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      // setRandomWeights( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );
      firstLayerWeightMatrix[0] = 1.24737338; firstLayerWeightMatrix[1] = 1.58455078;
      firstLayerWeightMatrix[2] = 0.28295388; firstLayerWeightMatrix[3] = 1.32056292;
      firstLayerWeightMatrix[4] = 0.69207227; firstLayerWeightMatrix[5] = -0.69103982;

      // setRandomWeights( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      secondLayerWeightVector[0] = -0.08738612;
      secondLayerWeightVector[1] = 0.23705916;
      secondLayerWeightVector[2] = 0.8396252;


      cout << "1st layer Weight Matrix" << endl;
      printMatrix( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );

      cout << "Output layer Weight Matrix" << endl;
      printMatrix( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      cout << "First Layer Calculations " << endl;
      // test MLP topology transformations

      initializeMatrix( a, NUM_HIDDEN_NODES , 1);
      initializeMatrix( z, NUM_HIDDEN_NODES , 1);

      computeActivations( x, firstLayerWeightMatrix, a);

      computeHiddenUnits( a , z, NUM_HIDDEN_NODES );
      
      cout << "Activations are:" << "\n";
      printMatrix( a, NUM_HIDDEN_NODES, 1);

      cout << "Hidden Units are:" << "\n";
      printMatrix( z, NUM_HIDDEN_NODES, 1);

      printf("-------------------------------------\n");

      //--------------------------------------------------------------------
      cout << "Second Layer Calculations " << endl;

      initializeMatrix( v, NUM_OUTPUTS , 1);

      computeOutputActivations( z, secondLayerWeightVector, v);
      cout << "Output activation is:" << "\n";
      printMatrix( v, NUM_OUTPUTS, 1);

      cout << "Prediction is:" << "\n";
      logisticSigmoid(v, y, NUM_OUTPUTS);
      printMatrix( y, NUM_OUTPUTS, 1);

      cout << "Forward Propagation Complete." << "\n";
      printf("-------------------------------------\n");

      cout << "Cross Entropy error is " << crossEntropyFunction( t , y ) << "\n";
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      cout << "Begin  BackPropagation... " << "\n";


      delta = computeOutputErrors( y, t);
      cout << "Output Errors " << delta  << "\n";


      initializeMatrix( hiddenDeltas, NUM_HIDDEN_NODES, 1);
      computeHiddenErrors( a, secondLayerWeightVector, delta, hiddenDeltas);
      printMatrix( hiddenDeltas, NUM_HIDDEN_NODES, 1);


      initializeMatrix( secondLayerDerivatives, NUM_OUTPUTS, NUM_HIDDEN_NODES );
      computeSecondLayerDerivatives( z, delta, secondLayerDerivatives);
      cout << "Second Layer derivatives " << endl;
      printMatrix( secondLayerDerivatives, NUM_HIDDEN_NODES, NUM_OUTPUTS);


      initializeMatrix( firstLayerDerivatives, NUM_HIDDEN_NODES, NUM_FEATURES );
      computeFirstLayerDerivatives( x, hiddenDeltas, firstLayerDerivatives);
      cout << "First Layer derivatives " << endl;
      printMatrix( firstLayerDerivatives, NUM_HIDDEN_NODES, NUM_FEATURES );

      printf("-------------------------------------\n");
      cout << "Update Parameters... " << "\n";

      //update 1st layer:
      updateFirstLayerWeights(firstLayerWeightMatrix, firstLayerDerivatives, LEARNING_RATE);

      //update 2nd layer:
      updateSecondLayerWeights(secondLayerWeightVector, secondLayerDerivatives, LEARNING_RATE);

      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      return 0;
}
