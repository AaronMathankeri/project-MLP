#include <iostream>
#include "mkl.h"
#include "mathimf.h"
#include <vector>

#include "feedForwardFunctions.hpp"
#include "ioFunctions.hpp"
#include "initializations.hpp"
#include "networkAgnosticFunctions.hpp"
#include "backpropFunctions.hpp"
#include "gradientDescent.hpp"

using namespace std;

int main(int argc, char *argv[])
{
      cout << " CREATING A SIMPLE NEURAL NET" << endl;
      printf("-------------------------------------\n");
      //--------------------------------------------------------------------
      float * x1 = (float *)mkl_malloc( NUM_FEATURES*sizeof( float ), 64 ); //features
      float * x2 = (float *)mkl_malloc( NUM_FEATURES*sizeof( float ), 64 ); //features
      float * firstLayerWeightMatrix = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * secondLayerWeightVector = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float * t1 = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 ); //targets
      float * t2 = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 ); //targets
      float * y = (float *)mkl_malloc( NUM_SAMPLES*sizeof( float ), 64 ); //predictions
      float * a = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );//activations
      float * z = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 );//hidden units
      float * v = (float *)mkl_malloc( NUM_OUTPUTS*sizeof( float ), 64 );//output activations
      float * hiddenDeltas = (float *)mkl_malloc( NUM_HIDDEN_NODES*sizeof( float ), 64 ); //errors from backprop
      float * firstLayerDerivatives = (float *)mkl_malloc( NUM_HIDDEN_NODES* NUM_FEATURES *sizeof( float ), 64 );
      float * secondLayerDerivatives = (float *)mkl_malloc( NUM_OUTPUTS*NUM_HIDDEN_NODES* sizeof( float ), 64 );
      float delta = 0.0; //error from initialbackprop

      initializeMatrix( x1, NUM_FEATURES , 1);
      initializeMatrix( t1, NUM_SAMPLES , 1);
      initializeMatrix( x2, NUM_FEATURES , 1);
      initializeMatrix( t2, NUM_SAMPLES , 1);

      initializeMatrix( y, NUM_SAMPLES , 1);
      //--------------------------------------------------------------------
      //hard-code some values
      x1[0] = 0.22455898;
      x1[1] = 0.56056104;
      x2[0] = 1.01417185;
      x2[1] = 0.01460828;

      t1[0] = 1.0;
      t2[0] = 0.0;

      vector<float*> inputData; 
      inputData.push_back(x1);
      inputData.push_back(x2);

      vector<float*> outputData; 
      outputData.push_back(t1);
      outputData.push_back(t2);

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

      initializeMatrix( firstLayerDerivatives, NUM_HIDDEN_NODES, NUM_FEATURES );
      initializeMatrix( secondLayerDerivatives, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      cout << "1st layer Weight Matrix" << endl;
      printMatrix( firstLayerWeightMatrix, NUM_HIDDEN_NODES, NUM_FEATURES );

      cout << "Output layer Weight Matrix" << endl;
      printMatrix( secondLayerWeightVector, NUM_OUTPUTS, NUM_HIDDEN_NODES );

      printf("-------------------------------------\n");
      double error = 0.0;
      for (int i = 0; i < 2; ++i) {

	    float* currentInput = inputData[i];
	    float* currentOutput = outputData[i];

	    cout << "Features are :" << endl;
	    printMatrix( currentInput, NUM_FEATURES, 1 );

	    cout << "Targets are :" << endl;
	    printMatrix( currentOutput, NUM_SAMPLES, 1 );

	    //printf("-------------------------------------\n");

	    //--------------------------------------------------------------------
	    // this needs to be done for each data sample!
	    cout << "First Layer Calculations " << endl;
	    // test MLP topology transformations

	    initializeMatrix( a, NUM_HIDDEN_NODES , 1);
	    initializeMatrix( z, NUM_HIDDEN_NODES , 1);

	    computeActivations( currentInput, firstLayerWeightMatrix, a);

	    computeHiddenUnits( a , z, NUM_HIDDEN_NODES );
      
	    cout << "Activations are:" << "\n";
	    printMatrix( a, NUM_HIDDEN_NODES, 1);

	    cout << "Hidden Units are:" << "\n";
	    printMatrix( z, NUM_HIDDEN_NODES, 1);

	    //printf("-------------------------------------\n");

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
	    //printf("-------------------------------------\n");
	    error += crossEntropyFunction( currentOutput, y );
	    cout << "Cross Entropy error is " << error << "\n";
	    //cout << "Cross Entropy error is " << crossEntropyFunction( currentOutput , y ) << "\n";
	    //printf("-------------------------------------\n");
	    //--------------------------------------------------------------------
	    cout << "Begin  BackPropagation... " << "\n";

	    delta = computeOutputErrors( y, currentOutput);
	    cout << "Output Errors " << delta  << "\n";

	    initializeMatrix( hiddenDeltas, NUM_HIDDEN_NODES, 1);
	    computeHiddenErrors( a, secondLayerWeightVector, delta, hiddenDeltas);
	    printMatrix( hiddenDeltas, NUM_HIDDEN_NODES, 1);

	    computeSecondLayerDerivatives( z, delta, secondLayerDerivatives);
	    cout << "Second Layer derivatives " << endl;
	    printMatrix( secondLayerDerivatives, NUM_HIDDEN_NODES, NUM_OUTPUTS);

	    computeFirstLayerDerivatives( currentInput, hiddenDeltas, firstLayerDerivatives);
	    cout << "First Layer derivatives " << endl;
	    printMatrix( firstLayerDerivatives, NUM_HIDDEN_NODES, NUM_FEATURES );
	    printf("-------------------------------------\n");
      }
      /*
      printf("-------------------------------------\n");
      cout << "Update Parameters... " << "\n";
      //update 1st layer:
      updateFirstLayerWeights(firstLayerWeightMatrix, firstLayerDerivatives, LEARNING_RATE);
      //update 2nd layer:
      updateSecondLayerWeights(secondLayerWeightVector, secondLayerDerivatives, LEARNING_RATE);
      printf("-------------------------------------\n");

      //*/

      //--------------------------------------------------------------------
      return 0;
}
