/**
 *   \file gradientDescent.cpp
 *   \brief the source
 *
 *  Detailed description
 *
 */
#include "../include/gradientDescent.hpp"

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
